from cogarch_tools.cogarch_run import main


def test_vlm_tamp_domain():
    main(
        problem='test_kitchen_chicken_soup', exp_subdir='kitchen_gpt', use_rel_pose=True,
        # observation_model='exposed'
    )


if __name__ == '__main__':
    test_vlm_tamp_domain()
