import json
import pickle
from os.path import join, isdir, isfile
from os import listdir
from collections import defaultdict

from world_builder.paths import DATABASES_PATH


pose_preds = {
    'supported': ([1, 3], 2),
    'contained': ([1, 3], 2),
}

position_preds = {
    'issampledposition': ([1, 2], 3)
}


def var(x):
    if '=' in x:
        return eval(x.split('=')[1])
    return x


def unpack(fact):
    name = fact[0].lower()
    if name in ['KinPullOneAction'.lower()]:
        _, a, o, p1, p2, g, bq1, bq2, bt, aq2, at = fact
        return [[a, o, p1, g, bq1], [a, o, p2, g, bq2]]
    else:
        _, a, o, p, g, q = fact[:6]
        p = var(p)
        g = var(g)
        q = var(q)
        return a, o, p, g, q


def save_sampler_database(dic, name):
    with open(join(DATABASES_PATH, f'{name}.pickle'), 'wb') as f:
        pickle.dump(dic, f)


def summarize_reachability(time_json_files, prefix='nvidia_kitchen_'):
    """ exp_runs : {} """

    or_to_p = defaultdict(list)  ## (movable, surface) to pose
    aopg_to_q = defaultdict(list)  ## IR for starting poses
    aog_to_p_to_q = {}  ## IR for sampled poses
    j_to_p = defaultdict(list)  ## joint to position
    ajg_to_p_to_q = {}  ## IR for sampled positions
    done = 0
    skipped = 0
    for f in time_json_files:
        for dic in json.load(open(f, 'r')):
            if 'init' not in dic:
                skipped += 1
                continue
            done += 1
            facts = dic['init']

            ## placing on / in
            sampled_poses = {f[2]: (f[1], f[-1]) for f in facts if f[0] in pose_preds}
            for p, o_r in sampled_poses.items():
                reach = [f for f in facts if f[0] == 'reach' and f[3] == p][0]
                a, o, p, g, q = unpack(reach)
                key = (a, o, g)
                if key not in aog_to_p_to_q:
                    aog_to_p_to_q[key] = defaultdict(list)
                aog_to_p_to_q[key][p].append(q)
                or_to_p[o_r].append(p)

            ## picking
            initial_poses = {f[3]: f for f in facts if f[0] == "kin" and f[3] not in sampled_poses}
            for p, kin in initial_poses.items():
                a, o, p, g, q = unpack(kin)
                # aopg_to_q[(a, o, p, g)].append(q)
                key = (a, o, g)
                if key not in aog_to_p_to_q:
                    aog_to_p_to_q[key] = defaultdict(list)
                aog_to_p_to_q[key][p].append(q)

            ## pulling door, new version has 'KinPullOneAction', older version has 'ungrasphandle' & 'grasphandle'
            sampled_positions = {f[-1]: (f[1], f[2]) for f in facts if f[0] in position_preds}
            for p, j_p in sampled_positions.items():
                j, p1 = j_p
                p1_var = var(p1)
                j_to_p[(j, p1_var)].append(var(p))

                ## newer domain for pulling action
                found_pullone = [f for f in facts if f[0].lower() == 'KinPullOneAction'.lower() and f[4] == p]
                if len(found_pullone) > 0:
                    pullone = found_pullone[0]
                    for a, j, p, g, q in unpack(pullone):
                        key = (a, j, g)
                        if key not in ajg_to_p_to_q:
                            ajg_to_p_to_q[key] = defaultdict(list)
                        ajg_to_p_to_q[key][p].append(q)

                ## older domain for pulling action
                found_ungrasphandle = [f for f in facts if f[0] == 'ungrasphandle' and f[3] == p]
                if len(found_ungrasphandle) > 0:
                    ungrasphandle = found_ungrasphandle[0]
                    a, j, p, g, q = unpack(ungrasphandle)
                    key = (a, j, g)
                    if key not in ajg_to_p_to_q:
                        ajg_to_p_to_q[key] = defaultdict(list)
                    ajg_to_p_to_q[key][p].append(q)

                    grasphandle = [f for f in facts if f[0] == 'grasphandle' and f[3] == p1][0]
                    q = unpack(grasphandle)[-1]
                    ajg_to_p_to_q[key][p1_var].append(q)

    print(f'reviewed {done} runs and skipped {skipped} runs')
    ajg_to_p_to_q = {k: dict(sorted(v.items(), key=lambda x: x[0])) for k, v in ajg_to_p_to_q.items()}

    save_sampler_database(ajg_to_p_to_q, f"{prefix}ajg_to_p_to_q")
    save_sampler_database(or_to_p, f"{prefix}or_to_p")
    # save_sampler_database(aopg_to_q, f"{prefix}aopg_to_q")
    save_sampler_database(aog_to_p_to_q, f"{prefix}aog_to_p_to_q")
    save_sampler_database(j_to_p, f"{prefix}j_to_p")
