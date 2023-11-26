import os
import time
from os.path import abspath


def show_rgb(rgba_matrix):
    from PIL import Image
    image = Image.fromarray(rgba_matrix, 'RGBA')
    image.show()


def save_rgb_jpg(rgba_matrix, jpg_path='observation.jpg'):
    from PIL import Image
    image = Image.fromarray(rgba_matrix, 'RGBA')
    image_rgb = image.convert('RGB')
    image_rgb.save(jpg_path)


def query_gpt4v(rgba_matrix, prompt="Describe all objects in this image exhaustively.", jpg_path='observation.jpg'):
    import requests
    from pprint import pprint

    start_time = time.time()
    kwargs = dict()
    payload = {"prompt": f"Answer in a python list: {prompt}"}

    # debugging uploading image
    if False:
        jpg_path = "https://www.reuters.com/resizer/NLk9k89J1tfmH-B7XKd598-6j_Y=/960x0/filters:quality(80)/cloudfront-us-east-2.images.arcpublishing.com/reuters/AHF2FYISNJO55J6N35YJBZ2JYY.jpg"
        jpg_path = "https://drive.google.com/file/d/1OcxQUuEsVbHbW7teJvRFXhbnCP2tSyCR/view"
        payload["image_url"] = jpg_path
        kwargs['json'] = payload

    else:
        save_rgb_jpg(rgba_matrix, jpg_path=jpg_path)
        payload["image_path"] = jpg_path
        kwargs['json'] = payload

    print('='*30)
    print('\n\nQuerying GPT-4V ...', payload['prompt'])
    response = requests.post(f"http://localhost:8000/action", **kwargs).json()

    # if 'http' not in jpg_path:
    #     os.remove(jpg_path)

    if 'status' in response and response['status'] == 'Success':
        response = response['result']['answer']
        print('Answer:', response)
    else:
        pprint(response)

    print(f'Done in {round(time.time() - start_time, 3)} sec\n\n')
    print('='*30)
    return response
