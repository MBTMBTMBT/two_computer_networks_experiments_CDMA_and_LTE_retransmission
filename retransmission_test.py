# authors: Benteng Ma and Guangzhao Cao
# 2020.6

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
ERR_LTE = 0.05
ERR_ROUTER = 0.005
TAO = 1
ERR_LTE_LIST = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# MIU_LTE = 0.1
# THETA_LTE = 1
# MIU_ROUTER = 0.05
# THETA_ROUTER = 0.5


# you may see that this is only test1, we originally want to have four tests,
# but for now we messed up with these names, please don't mind them
def test1(err1: float, err2: float):
    # test LTE without retransmission
    # for different number of nodes
    # for nodes in range(5, 31):

    node_nums = []
    frame_nums = []
    transmission_counts = []
    for nodes in range(5, 6):
        node_nums.append(nodes)

        # for different number of frames
        # for frame_num_pwr in range(5, 11):
        for frame_num_pwr in range(1, 8):
            frame_num = 10 ** frame_num_pwr  # let's test with increase of power
            frame_nums.append(frame_num)

            # start of each test
            transmission_count = 0

            # get error rates
            # in this test, the receiver error is different from routes
            # error_rates = [np.random.normal(MIU_LTE, THETA_LTE, frame_num).reshape((1, frame_num))]
            error_rates = [np.ones((1, frame_num)) * err1]
            # add error vector for each router
            for each_node in range(nodes):
                # error_rates.append(np.random.normal(MIU_ROUTE, THETA_ROUTE, frame_num).reshape(1, frame_num))
                error_rates.append(np.ones((1, frame_num)) * err2)

            # start transmission
            # print("start transmission")

            success_flag = np.zeros((1, frame_num))  # to see if this frame already arrives
            success_flag = np.array(success_flag, dtype=bool)  # again turn this into booleans

            while True:
                node_count = 1
                # create frames or reset frames
                frames = np.ones((1, frame_num))  # first create ones with this shape
                frames = np.array(frames, dtype=bool)  # then turn them into booleans

                # try to loss some frames
                for each_rate in error_rates:
                    # np.random.seed(np.random.randint(0, 1000))
                    # print("Node: %d" % node_count)
                    node_count += 1
                    error_rand = np.random.rand(1, frame_num)
                    success = error_rand > each_rate
                    # print(success)
                    for i in range(success.shape[1]):
                        if success_flag[:, i]:
                            continue
                        frames[:, i] = frames[:, i] and success[:, i]
                        transmission_count += 1

                # if a frame is successfully transmitted for ones,
                # it is successful forever
                for i in range(success_flag.shape[1]):
                    success_flag[:, i] = success_flag[:, i] or frames[:, i]
                # print(success_flag)

                # if all the frames transmitted, it is successful
                escape_flag = False
                if np.all(success_flag):
                    transmission_counts.append(transmission_count)
                    break

    return node_nums, frame_nums, transmission_counts


def test1_with_retransmission(err1: float, err2: float):
    # test LTE without retransmission
    # for different number of nodes
    # for nodes in range(5, 31):

    node_nums = []
    frame_nums = []
    transmission_counts = []
    for nodes in range(5, 6):
        node_nums.append(nodes)

        # for different number of frames
        # for frame_num_pwr in range(5, 11):
        for frame_num_pwr in range(1, 8):
            frame_num = 10 ** frame_num_pwr  # let's test with increase of power
            frame_nums.append(frame_num)

            # start of each test
            transmission_count = 0

            # get error rates
            # in this test, the receiver error is different from routes
            # error_rates = [np.random.normal(MIU_LTE, THETA_LTE, frame_num).reshape((1, frame_num))]
            error_rates = [np.ones((1, frame_num)) * err1]
            # add error vector for each router
            for each_node in range(nodes):
                # error_rates.append(np.random.normal(MIU_ROUTE, THETA_ROUTE, frame_num).reshape(1, frame_num))
                error_rates.append(np.ones((1, frame_num)) * err2)

            # start transmission
            # print("start transmission")

            success_flag = np.zeros((1, frame_num))  # to see if this frame already arrives
            success_flag = np.array(success_flag, dtype=bool)  # again turn this into booleans

            while True:
                node_count = 1
                # create frames or reset frames
                frames = np.ones((1, frame_num))  # first create ones with this shape
                frames = np.array(frames, dtype=bool)  # then turn them into booleans

                # try to loss some frames
                first_flag = True
                for each_rate in error_rates:
                    # np.random.seed(np.random.randint(0, 1000))
                    # print("Node: %d" % node_count)
                    node_count += 1
                    error_rand = np.random.rand(1, frame_num)
                    success = error_rand > each_rate
                    # print(success)
                    for i in range(success.shape[1]):
                        if success_flag[:, i]:
                            continue
                        frames[:, i] = frames[:, i] and success[:, i]
                        transmission_count += 1

                    # if this is the first node, then it is a LTE cell, and we will consider retransmission here
                    if first_flag:
                        first_flag = False

                        # do retransmission, until all the frames success
                        # all_correct = False
                        '''
                        frames_temp = frames.copy()
                        while not frames_temp.all():
                            error_rand = np.random.rand(1, frame_num)
                            success = error_rand > each_rate
                            for i in range(success.shape[1]):
                                if frames_temp[:, i]:
                                    continue
                                frames[:, i] = frames[:, i] or success[:, i]
                                if frames[:, 1]:
                                    frames_temp[:, 1] = True
                                transmission_count += 1
                            print(frames_temp)
                        frames = frames_temp  # all should finally success
                        '''

                        # do retransmission, until all the frames success
                        for i in range(frames.shape[1]):
                            if not frames[:, i]:
                                frames[:, i] = True
                                transmission_count += 1

                # if a frame is successfully transmitted for ones,
                # it is successful forever
                for i in range(success_flag.shape[1]):
                    success_flag[:, i] = success_flag[:, i] or frames[:, i]
                # print(success_flag)

                # if all the frames transmitted, it is successful
                escape_flag = False
                if np.all(success_flag):
                    transmission_counts.append(transmission_count)
                    break

    return node_nums, frame_nums, transmission_counts


def test1_time_and_flow(err1: float, err2: float, tao):
    # test LTE without retransmission
    # for different number of nodes
    # for nodes in range(5, 31):

    node_nums = []
    frame_nums = []
    transmission_counts = []
    time_counts = []
    for nodes in range(5, 6):
        node_nums.append(nodes)

        # for different number of frames
        # for frame_num_pwr in range(5, 11):
        for frame_num_pwr in range(1, 8):
            frame_num = 10 ** frame_num_pwr  # let's test with increase of power
            frame_nums.append(frame_num)

            # start of each test
            transmission_count = 0
            transmitted_count = 0
            time_count = 0

            # get error rates
            # in this test, the receiver error is different from routes
            # error_rates = [np.random.normal(MIU_LTE, THETA_LTE, frame_num).reshape((1, frame_num))]
            error_rates = [np.ones((1, frame_num)) * err1]
            # add error vector for each router
            for each_node in range(nodes):
                # error_rates.append(np.random.normal(MIU_ROUTE, THETA_ROUTE, frame_num).reshape(1, frame_num))
                error_rates.append(np.ones((1, frame_num)) * err2)

            # start transmission
            # print("start transmission")

            success_flag = np.zeros((1, frame_num))  # to see if this frame already arrives
            success_flag = np.array(success_flag, dtype=bool)  # again turn this into booleans

            while True:
                node_count = 1
                # create frames or reset frames
                frames = np.ones((1, frame_num))  # first create ones with this shape
                frames = np.array(frames, dtype=bool)  # then turn them into booleans

                # try to loss some frames
                for each_rate in error_rates:
                    # np.random.seed(np.random.randint(0, 1000))
                    # print("Node: %d" % node_count)
                    node_count += 1
                    error_rand = np.random.rand(1, frame_num)
                    success = error_rand > each_rate
                    # print(success)
                    for i in range(success.shape[1]):
                        if success_flag[:, i]:
                            continue
                        frames[:, i] = frames[:, i] and success[:, i]
                        transmission_count += 1

                # compute time
                time_count += (frame_num - transmitted_count + nodes) * tao

                # if a frame is successfully transmitted for ones,
                # it is successful forever
                for i in range(success_flag.shape[1]):
                    success_flag[:, i] = success_flag[:, i] or frames[:, i]
                # print(success_flag)

                # count for success
                transmitted_count = 0
                for i in range(success_flag.shape[1]):
                    if success_flag[:, i]:
                        transmitted_count += 1

                # if all the frames transmitted, it is successful
                if np.all(success_flag):
                    transmission_counts.append(transmission_count)
                    time_counts.append(time_count)
                    break

    return node_nums, frame_nums, transmission_counts, time_counts


def test1_with_different_rates(err1: [], err2: float):
    # test LTE without retransmission
    # for different number of nodes
    # for nodes in range(5, 31):

    node_nums = []
    frame_nums = []
    transmission_counts = []
    for each_err in err1:
        for nodes in range(5, 6):
            node_nums.append(nodes)

            # for different number of frames
            # for frame_num_pwr in range(5, 11):
            for frame_num_pwr in range(7, 8):
                frame_num = 10 ** frame_num_pwr  # let's test with increase of power
                frame_nums.append(frame_num)

                # start of each test
                transmission_count = 0

                # get error rates
                # in this test, the receiver error is different from routes
                # error_rates = [np.random.normal(MIU_LTE, THETA_LTE, frame_num).reshape((1, frame_num))]
                error_rates = [np.ones((1, frame_num)) * each_err]
                # add error vector for each router
                for each_node in range(nodes):
                    # error_rates.append(np.random.normal(MIU_ROUTE, THETA_ROUTE, frame_num).reshape(1, frame_num))
                    error_rates.append(np.ones((1, frame_num)) * err2)

                # start transmission
                # print("start transmission")

                success_flag = np.zeros((1, frame_num))  # to see if this frame already arrives
                success_flag = np.array(success_flag, dtype=bool)  # again turn this into booleans

                while True:
                    node_count = 1
                    # create frames or reset frames
                    frames = np.ones((1, frame_num))  # first create ones with this shape
                    frames = np.array(frames, dtype=bool)  # then turn them into booleans

                    # try to loss some frames
                    for each_rate in error_rates:
                        # np.random.seed(np.random.randint(0, 1000))
                        # print("Node: %d" % node_count)
                        node_count += 1
                        error_rand = np.random.rand(1, frame_num)
                        success = error_rand > each_rate
                        # print(success)
                        for i in range(success.shape[1]):
                            if success_flag[:, i]:
                                continue
                            frames[:, i] = frames[:, i] and success[:, i]
                            transmission_count += 1

                    # if a frame is successfully transmitted for ones,
                    # it is successful forever
                    for i in range(success_flag.shape[1]):
                        success_flag[:, i] = success_flag[:, i] or frames[:, i]
                    # print(success_flag)

                    # if all the frames transmitted, it is successful
                    escape_flag = False
                    if np.all(success_flag):
                        transmission_counts.append(transmission_count)
                        break

    return node_nums, frame_nums, transmission_counts


if __name__ == '__main__':
    '''
    node_nums1, frame_nums1, transmission_counts1 = test1(ERR_LTE, ERR_ROUTER)
    print(node_nums1, frame_nums1, transmission_counts1)
    node_nums2, frame_nums2, transmission_counts2 = test1(ERR_ROUTER, ERR_ROUTER)
    print(node_nums2, frame_nums2, transmission_counts2)

    # frame_nums1 = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    # transmission_counts1 = [121, 1265, 13046, 126973, 1277628, 12805023, 128038185]
    # frame_nums2 = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    # transmission_counts2 = [121, 1331, 12067, 123354, 1228249, 12280939, 122867525]

    # frame_nums1 = np.log10(frame_nums1)
    # frame_nums2 = np.log10(frame_nums2)
    # transmission_counts1 = np.log10(transmission_counts1)
    # transmission_counts2 = np.log10(transmission_counts2)

    plt.figure()
    plt.title("number of frames / times of transmission")
    plt.xlabel("number of frames")
    plt.ylabel("times of transmission")
    plt.plot(frame_nums1, transmission_counts1, color="deeppink", linestyle=":", label="LTE", marker="o")
    plt.plot(frame_nums2, transmission_counts2, color="darkblue", linestyle="--", label="wired systems", marker="x")
    plt.legend()
    plt.show()
    '''

    '''
    node_nums1, frame_nums1, transmission_counts1, time_counts1 = test1_time_and_flow(ERR_LTE, ERR_ROUTER, TAO)
    print(node_nums1, frame_nums1, transmission_counts1)
    node_nums2, frame_nums2, transmission_counts2, time_counts2 = test1_time_and_flow(ERR_ROUTER, ERR_ROUTER, TAO)
    print(node_nums2, frame_nums2, transmission_counts2)

    plt.figure()
    plt.title("number of frames / time length of transmission")
    plt.xlabel("number of frames")
    plt.ylabel("time length of transmission")
    plt.plot(frame_nums1, time_counts1, color="deeppink", linestyle=":", label="LTE", marker="o")
    plt.plot(frame_nums2, time_counts2, color="darkblue", linestyle="--", label="wired systems", marker="x")
    plt.legend()
    plt.show()
    '''

    '''
    node_nums, frame_nums, transmission_counts = test1_with_different_rates(ERR_LTE_LIST, ERR_ROUTER)
    plt.figure()
    plt.title("error rates / times of transmission")
    plt.xlabel("error rate")
    plt.ylabel("times of transmission")
    plt.plot(ERR_LTE_LIST, transmission_counts, color="deeppink", linestyle=":", label="LTE", marker="o")
    plt.legend()
    plt.show()
    '''

    node_nums0, frame_nums0, transmission_counts0 = test1(0, 0)
    print(node_nums0, frame_nums0, transmission_counts0)
    node_nums1, frame_nums1, transmission_counts1 = test1(ERR_LTE, ERR_ROUTER)
    print(node_nums1, frame_nums1, transmission_counts1)
    node_nums2, frame_nums2, transmission_counts2 = test1(ERR_ROUTER, ERR_ROUTER)
    print(node_nums2, frame_nums2, transmission_counts2)
    node_nums3, frame_nums3, transmission_counts3 = test1_with_retransmission(ERR_ROUTER, ERR_ROUTER)
    print(node_nums3, frame_nums3, transmission_counts3)

    plt.figure()
    plt.title("number of frames / times of transmission")
    plt.xlabel("number of frames")
    plt.ylabel("times of transmission")
    plt.plot(frame_nums0, transmission_counts0, color="gray", linestyle="-", label="no error", marker=".")
    plt.plot(frame_nums1, transmission_counts1, color="deeppink", linestyle=":", label="LTE", marker="o")
    plt.plot(frame_nums2, transmission_counts2, color="darkblue", linestyle="--", label="wired systems", marker="x")
    plt.plot(frame_nums3, transmission_counts3, color="orange", linestyle="-.", label="LTE_with_retransmission", marker="+")
    plt.legend()
    plt.show()
