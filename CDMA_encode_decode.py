# authors: Benteng Ma and Guangzhao Cao
# 2020.6

import numpy as np
import matplotlib.pyplot as plt


# borrowed from https://blog.csdn.net/dog250/article/details/6420427
CODE_A = np.array([1, 1, 1, 1, 1, 1, 1, 1])
CODE_B = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
CODE_C = np.array([-1, -1, 1, 1, 1, 1, -1, -1])
CODE_D = np.array([1, 1, -1, -1, 1, 1, -1, -1])
CODE_E = np.array([1, -1, -1, 1, 1, -1, -1, 1])
CODE_F = np.array([-1, 1, 1, -1, 1, -1, -1, 1])
CODE_G = np.array([-1, 1, -1, 1, 1, -1, 1, -1])
CODE_H = np.array([1, -1, 1, -1, 1, -1, 1, -1])
CODES = [CODE_A, CODE_B, CODE_C, CODE_D, CODE_E, CODE_F, CODE_G, CODE_H]


def get_unicode(string: str):
    unicode_lst = []
    for each in string:
        unicode_lst.append(ord(each))
    return unicode_lst


def get_eight_bit_bin(unicode_lst: []):
    bin_rst = []
    for i in unicode_lst:
        # method from https://blog.csdn.net/iteslafan/article/details/101122471
        bin_rst.append([i >> d & 1 for d in range(8)][::-1])
    return bin_rst

def get_str(unicode_lst: []):
    rst = ""
    for each_code in unicode_lst:
        rst += chr(each_code)
    return rst

def get_decimal(bin_lst):
    dec_lst = []
    for i in range(len(bin_lst) // 8):
        bin_ascii = bin_lst[i*8: i*8+8]
        bin_str = ""
        for each_bit in bin_ascii:
            bin_str += str(each_bit)
        dec = int(bin_str, 2)  # method from https://blog.csdn.net/pyufftj/article/details/72085698
        dec_lst.append(dec)
    return dec_lst


def encode(codes: [], strings: []):

    # encoded info for each channel
    encoded_lst = []

    for channel_count in range(len(strings)):
        # first encode these strings into ascii (8 bits, so ascii)
        unicode_lst = get_unicode(strings[channel_count])

        # turn these into binary
        bin_lst = get_eight_bit_bin(unicode_lst)

        # get the code for this channel
        code = codes[channel_count]

        each_encoded = []
        for each_bin in bin_lst:
            for each_bin_bit in each_bin:
                if each_bin_bit == 1:
                    for each_bit in code:
                        each_encoded.append(each_bit)
                elif each_bin_bit == 0:
                    for each_bit in code:
                        each_encoded.append(-each_bit)
        encoded_lst.append(each_encoded)

    # find the longest message
    if len(encoded_lst) == 0:
        return np.array([])
    longest = encoded_lst[0]
    for each_encoded in encoded_lst:
        if len(longest) < len(each_encoded):
            longest = each_encoded
    max_length = len(longest)
    for each_encoded in encoded_lst:
        while max_length - len(each_encoded) > 0:
            each_encoded.append(0)  # add zeros to get the same in length

    # finally, make them overlapped
    rst = np.zeros((1, max_length))
    for each_encoded in encoded_lst:
        rst += np.array(each_encoded).reshape((1, max_length))

    return rst, encoded_lst


def decode(codes: [], encoded_lst_np: np.array):
    # encoded_lst = encoded_lst_py.tolist()
    encoded_lst_eight_bits_np = []
    for i in range(encoded_lst_np.shape[1] // 8):
        each_eight_bits_np = encoded_lst_np[0, i * 8: i * 8 + 8].reshape((1, 8))
        encoded_lst_eight_bits_np.append(each_eight_bits_np)

    strings = []
    for each_code in codes:
        code_np = np.transpose(np.array(each_code).reshape((1, len(each_code))))
        bin_rst_lst_channel = []
        for each_eight_bits_np in encoded_lst_eight_bits_np:
            dot_rst = np.dot(each_eight_bits_np, code_np)
            if dot_rst > 0:
                bin_rst = 1
            else:
                bin_rst = 0
            bin_rst_lst_channel.append(bin_rst)
        dec_rst_lst_channel = get_decimal(bin_rst_lst_channel)
        channel_str = get_str(dec_rst_lst_channel)
        strings.append(channel_str)
    return strings


if __name__ == '__main__':
    info_lst = []
    info_lst.append("Hello world!")
    info_lst.append("This is a network encoding experiment!")
    info_lst.append("We're MBT and CGZ,")
    info_lst.append("we gonna try for CDMA.")
    info_lst.append("")
    info_lst.append("Above is an empty line,")
    info_lst.append("we still have one channel to go,")
    info_lst.append("the last channel.")
    encoded, encoded_lst = encode(CODES, info_lst)
    # print(encoded)

    plt.figure()
    plt.title("8-channel encoded CDMA")
    plt.xlabel("bits")
    plt.ylabel("volts")
    plt.plot(range(encoded.shape[1]), (encoded.reshape((encoded.shape[1]))).tolist(), color="deeppink", linestyle=":", label="mixed", marker="o")
    count = 0
    for each in encoded_lst:
        plt.plot(range(encoded.shape[1]), each, linestyle="--", label="channel_%d" % count, marker="x")
        count += 1
    plt.legend()
    plt.show()

    strings = decode(CODES, encoded)
    for each_str in strings:
        print(each_str)
