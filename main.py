import pandas as pd
import numpy as np
import os

## Function Definitions:

def convert_twos_complement_to_decimal(binary_str):
    # Determine the length of the binary string
    n = len(binary_str)

    # Convert the binary string to a list of numeric values
    binary_vector = [int(bit) for bit in binary_str]

    # Check if the number is negative by looking at the most significant bit
    is_negative = binary_vector[0] == 1

    if is_negative:
        # If negative, calculate the decimal value using two's complement conversion
        # Invert the bits
        inverted_bits = [1 - bit for bit in binary_vector]

        # Add 1 to the least significant bit
        carry = 1
        for i in range(n - 1, -1, -1):
            inverted_bits[i] += carry
            if inverted_bits[i] == 2:
                inverted_bits[i] = 0
                carry = 1
            else:
                carry = 0

        # Calculate the decimal value
        decimal_value = -sum(2**i for i, bit in enumerate(reversed(inverted_bits)) if bit == 1)
    else:
        # If positive, calculate the decimal value normally
        decimal_value = sum(2**i for i, bit in enumerate(reversed(binary_vector)) if bit == 1)

    return decimal_value

def convert_hex_to_decimal(hex_str, endian="big"):
    # Debug option to return the input hex string
    if endian == "debug":
        return hex_str

    # Check if the input is empty
    if not hex_str:
        return "EMPTY"

    # Check if the input is a valid hexadecimal string
    if not all(c in "0123456789abcdefABCDEF" for c in hex_str):
        return f"INVALID: {hex_str}"

    # Ensure the length of the hex string is even
    if len(hex_str) % 2 != 0:
        raise ValueError("Hexadecimal string length should be even.")

    # Split the hex string into bytes
    bytes_list = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]

    if endian == "little":
        # Reverse the byte order for little endian
        bytes_list = bytes_list[::-1]

    # Recombine the bytes into a single hex string
    big_endian_hex = "".join(bytes_list)

    # Convert the big endian hex string to binary
    binary_value = bin(int(big_endian_hex, 16))[2:].zfill(len(big_endian_hex) * 4)

    # Convert the binary value to a decimal number
    decimal_value = convert_twos_complement_to_decimal(binary_value)

    return decimal_value


# Reads in the file
with open("raw data/wavhex.txt", "r") as file:
    raw_hex = file.read()

# Gets rid of the meaningless 0x prefix before every 2-digit hex value and removes commas from the string.
# `hex_string` is now a clean hex string, ready to be processed.
no_prefix = raw_hex.replace("0x", "")
hex_string = "".join(no_prefix).replace(",", "")




