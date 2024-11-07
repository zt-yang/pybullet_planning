import json

import requests
import copy
import signal
import sys
import time
from pprint import pprint, pformat
import os
from os.path import isfile, abspath, join, isdir

from examples.pybullet.utils.pybullet_tools.utils import wait_for_user
from pybullet_tools.logging_utils import dump_json

from vlm_tools.vlm_utils import encode_image, is_interactive, cache_name
from vlm_tools.prompts_gpt4v import *

TOOL_DIR = abspath(join(__file__, '..'))
KEY_DIR = join(TOOL_DIR, 'keys')
MEMORY_KEYS = ['prompt', 'image_path', 'response', 'duration']


class VLMApi(object):

    name = 'VLM API wrapper'
    api_key = None
    model_name = None

    def __init__(self, image_dir: str = join(TOOL_DIR, '../assets/images'),
                 cache_dir: str = join(TOOL_DIR, 'cache/')):

        # self.prompt_templates = self.load_prompt_templates()
        self.image_dir = abspath(image_dir)
        self.last_image_path = None
        self.next_image_path = None

        self.cache_dir = abspath(cache_dir)
        if not isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.next_session_new = True
        self.context = []  ## list of API messages
        self.memory = {'is_loaded': False}  ## prompt_name: dict with keys (prompt, image_path, answer, query_time)
        self.query_time = []

    def new_session(self, image_dir=None, cache_dir=None):
        if image_dir is not None:
            self.image_dir = image_dir
        if cache_dir is not None:
            self.cache_dir = cache_dir
        self.next_session_new = True
        self.context = []
        self.memory = {}
        self.query_time = []

    # @staticmethod
    # def load_prompt_templates():
    #     return {
    #         'describe': prompt_describe,
    #         'where': prompt_where,
    #         'plan': prompt_plan,
    #     }
    #
    # def ask_by_template(self, template_name: str, prompt_kwargs: dict = dict(), **kwargs):
    #     prompt = self.prompt_templates[template_name]
    #     if len(prompt_kwargs) > 0:
    #         prompt = prompt.format(**prompt_kwargs)
    #     return self.ask(prompt, **kwargs)

    def _load_previous_image(self, display_image):
        image_path = self.next_image_path
        if self.next_image_path is not None:
            display_image = False
            self.next_image_path = None
        return image_path, display_image

    def ask(self, prompt: str, image_name=None, display_image=False, image_description='',
            prompt_name=None, image_dir=None, **kwargs):
        title = '\t[vlm_api.ask]'
        print(f'\n\n{title}\tquerying {self.name} ({self.model_name}) ...')
        start_time = time.time()

        ## ----------- image related -------------------------------------
        image_path, display_image = self._load_previous_image(display_image)

        if image_name is not None:
            # display_image = True
            if image_dir is not None:
                self.image_dir = image_dir
            image_path = join(self.image_dir, image_name)

        if image_path is not None:
            if isfile(image_path):
                print(f'{title}\tUsing image: {image_path}')
                prompt += image_description
                if display_image:
                    self.show_image(image_path)
            else:
                print(f'{title}\tWantimg to use image but not found: {image_path}')
                wait_for_user('Continue query without image?')

        ## ----------- query -------------------------------------

        print('-'*40 + '\n' + prompt + '\n' + '-'*40)
        continue_chat = not self.next_session_new
        response = self._ask(prompt, image_path, continue_chat=continue_chat, **kwargs)
        self.next_session_new = False

        ## ----------- process response -------------------------------------

        if isinstance(response, dict):
            pprint(response)
        else:
            print(response)

        duration = time.time() - start_time
        print(f'... in {round(duration, 3)} sec')

        self.query_time.append(duration)
        self.add_memory(prompt_name, (prompt, image_path, response, duration))
        self.save_memory()
        return response

    def _ask(self, prompt: str, image_path: str = "None", continue_chat: bool = True, **kwargs):
        raise NotImplementedError('should implement this for VLMAgent!')

    def add_memory(self, key, values):
        self.memory['is_loaded'] = False
        self.memory[key] = dict(zip(MEMORY_KEYS, values))

    def save_memory(self):
        cache_path = join(self.cache_dir, cache_name)
        with open(cache_path, 'w') as f:
            json.dump(self.memory, f, indent=3)
        print(f'\nVLMApi | Saved memory to {cache_path}\n')

    ########################################

    def show_last_image(self):
        if self.last_image_path is not None:
            self.show_image(self.last_image_path)

    def show_image(self, image_path=None, image_name=None):
        from PIL import Image
        if image_name is not None:
            image_path = join(self.image_dir, image_name)

        image = Image.open(image_path)
        if is_interactive():
            from IPython.display import display
            display(image)
        else:
            image.show()

    ########################################

    def save_context(self, session_name: str):
        session_path = join(self.cache_dir, session_name+'.txt')
        with open(session_path, 'w') as f:
            json.dump(self.context, f, indent=2)

    def load_context(self, session_name: str):
        session_path = join(self.cache_dir, session_name+'.txt')
        with open(session_path, 'r') as f:
            self.context = json.load(f)

    def save_agent(self, session_name: str):
        import pickle
        session_path = join(self.cache_dir, session_name+'.pkl')
        with open(session_path, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_agent(self, session_name: str):
        import pickle
        session_path = join(self.cache_dir, session_name+'.pkl')
        with open(session_path, 'rb') as f:
            self.__dict__ = pickle.load(f)
        self.next_image_path = self.last_image_path

    ########################################

    @staticmethod
    def load_api_key(api_key_txt_file, api_key_env_name):
        """ two ways to store the api key """

        key_file = join(KEY_DIR, api_key_txt_file)
        if isfile(key_file):
            with open(key_file, 'r') as f:
                api_key = f.read().strip()

        else:
            env_var = api_key_env_name
            if env_var not in os.environ:
                raise RuntimeError(f'Must set the environment variable `{env_var}`.'
                                   f'Please follow instructions in pybullet_planning/vlm_tools/README.md')
            api_key = os.environ[env_var]
        return api_key

    @staticmethod
    def requirements():
        raise NotImplementedError('should implement this for VLMAgent!')


################################################################################


class Claude3Api(VLMApi):

    name = 'Claude-3'

    def __init__(self, **kwargs):
        super(Claude3Api, self).__init__(**kwargs)

        self.api_key = self.load_api_key('claude_api_key.txt', 'ANTHROPIC_API_KEY')
        self.client = None
        self.messages = []
        self.model_name = "claude-3-opus-20240229"

    def _ask(self, prompt: str, image_path: str = "None", continue_chat: bool = True,
             system: str = "Respond with a python list",
             max_tokens: int = 1000, temperature: float = 0.0, n: int = 1, **kwargs):
        import anthropic

        if self.client is None or not continue_chat:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.messages = []

        content = prompt
        if image_path != "None" and image_path is not None:
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_path,
                    }
                },
                {"type": "text", "text": prompt}
            ]

        self.messages += [{"role": "user", "content": content}]

        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            temperature=temperature,
            system=system,
            messages=self.messages
        )
        self.messages += [{"role": "assistant", "content": message.content}]

        responses = [c.text for c in message.content]
        print('\n\n'.join(responses))
        if len(responses) == 1:
            return responses[0]
        return responses


class GPT4vApi(VLMApi):

    name = 'GPT-4v'

    def __init__(self, **kwargs):
        super(GPT4vApi, self).__init__(**kwargs)

        self.api_key = self.load_api_key('openai_api_key.txt', 'OPENAI_API_KEY')
        self.model_name = "gpt-4o-mini"  ## gpt-4o-mini | gpt-4o | gpt-4-vision-preview

    def _ask(self, prompt: str, image_path: str = "None", continue_chat: bool = True,
             max_tokens: int = 1000, temperature: float = 0.0, n: int = 1, **kwargs):

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        content = [{"type": "text", "text": prompt}]
        if image_path != "None" and image_path is not None:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                }
            })
            self.last_image_path = image_path

        messages = copy.deepcopy(self.context)
        messages.append({"role": "user", "content": content})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n
        }

        def timeout_handler(num, stack):
            print(f"setting timeout_handler for {num}")
            raise Exception("TIMEOUT")

        gpt_query_timeout = 10
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(gpt_query_timeout)
        import traceback
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions",
                                     headers=headers, json=payload).json()
        except Exception as ex:
            traceback.print_exc()
            print(f"\nTimed out GPT-query in {gpt_query_timeout} sec\n")
            sys.exit()
        finally:
            signal.alarm(0)

        if 'choices' not in response:
            print('\n\nChatGPT response\n')
            pprint(response)

        answers = [ans['message'] for ans in response['choices']]

        if "choices" in response:
            self.context.append({"role": "user", "content": prompt})
            self.context.extend(answers)

        contents = []
        for answer in answers:
            answer = answer['content']
            if answer.startswith('```json'):
                answer = answer.replace('\n', '').replace('```json{', '{').replace('}```', '}')
                answer = json.loads(answer)
            contents.append(answer)

        return contents


# ########################## FOR TESTING WHEN GPT-4V API is yet released ###############################
#
#
# class GPT4vOldApi(VLMApi):
#     """
#     1. install requirements
#         https://github.com/zt-yang/gpt4-image-api/tree/master
#     2. run the server `python main.py` inside the repo
#     """
#     PORT_NUMBER = 8000
#
#     def __init__(self):
#         super(GPT4vFreeApi, self).__init__()
#
#     def _ask(self, prompt: str, image_path: str = "None", continue_chat: bool = True, **kwargs):
#
#         payload = {
#             "continue_chat": continue_chat, "image_path": image_path, "prompt": prompt
#         }
#
#         response = requests.post(
#             f"http://localhost:{self.PORT_NUMBER}/action",
#             json=payload,
#         ).json()
#
#         if 'status' in response and response['status'] == 'Success':
#             response = response['result']
#             print('Answer:', response)
#         else:
#             pprint(response)
#
#         return response
#
#     @staticmethod
#     def requirements():
#         packages = ['fastapi==0.103.2',
#                     'python-dotenv==1.0.0',
#                     'undetected-chromedriver==3.5.3',
#                     'uvicorn==0.23.2']
#         return "pip install " + " ".join(packages)
