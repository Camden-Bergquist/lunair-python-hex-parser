## Packages.

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

def get_sample_size(header):
    """Return the sample size based on the header value."""
    # if-elif-else used instead of match statement because script is being written in pre-3.10 Python. (3.9.13)
    if header == "00":  # WV_TRANST_IMPED (2 bytes per sample).
        return 4
    elif header == "01":  # WV_THERAPY (2 bytes per sample).
        return 4
    elif header == "02":  # WV_IMU (12 bytes per sample).
        return 24
    elif header == "03":  # WV_DIGITAL_EVENTS (5 bytes per sample).
        return 10
    elif header == "FE":  # WV_FILLER (0 bytes per sample).
        return 0
    elif header == "FF":  # WV_NO_DATA (0 bytes per sample).
        return 0
    else:
        raise ValueError(f"Unknown Header: {header}")

def parse_to_packets(hex_string):
    """
    Converts a hex string into a list of substrings, each substring representing a single packet.
    
    Args:
        hex_string (str): The input hexadecimal string.

    Returns:
        list: A list of packet substrings.
    """
    # Initialize variables
    packets = []
    index = 0
    hex_length = len(hex_string)

    while index < hex_length:
        # Store starting index
        packet_start = index

        # Extract header (2 characters from the current index)
        header = hex_string[index:index + 2]

        # Handle filler values (header == "FE")
        if header == "FE":
            index += 2
        else:
            # Get sample size based on header
            sample_size = get_sample_size(header)

            # Skip to relative index 19 (18 characters ahead from the current index)
            index += 18

            # Extract the next four hexadecimal digits (19th to 22nd) for the number of samples
            num_samples_hex = hex_string[index:index + 4]
            index += 4

            # Convert the hexadecimal value to decimal
            num_samples = convert_hex_to_decimal(num_samples_hex, endian="little")

            # Skip the next twelve hexadecimal digits for the system timestamp and sequence number
            index += 12

            # Calculate the endpoint of the packet
            packet_end = index + (num_samples * sample_size)

            # Extract the packet
            packet = hex_string[packet_start:packet_end]
            packets.append(packet)

            # Update the starting index for the next packet
            index = packet_end

    return packets

def get_header_name(header):
    """
    Returns the name corresponding to the given header.

    Args:
        header (str): The header code.

    Returns:
        str: The name of the header.

    Raises:
        ValueError: If the header is unknown.
    """
    header_map = {
        "00": "WV_TRANST_IMPED",
        "01": "WV_THERAPY",
        "02": "WV_IMU",
        "03": "WV_DIGITAL_EVENTS",
        "FE": "WV_FILLER",
        "FF": "WV_NO_DATA",
    }

    if header not in header_map:
        raise ValueError("Unknown header")

    return header_map[header]

def parse_to_data_frame(hex_list):
    """
    Converts a list of hex packet substrings into a DataFrame with the decoded information.

    Args:
        hex_list (list): A list of hex strings representing packets.

    Returns:
        pd.DataFrame: A DataFrame containing the decoded packet information.
    """
    output_list = []

    # Iterate over the list of hex substrings
    for sub_string in hex_list:
        string_index = 0
        samples = []

        # Extract header and determine sample size and name
        header = sub_string[string_index:string_index + 2]
        sample_size = get_sample_size(header)
        header = get_header_name(header)
        string_index += 2

        # Extract RTC timestamp
        RTC_timestamp = convert_hex_to_decimal(sub_string[string_index:string_index + 8], "little")
        string_index += 16

        # Extract number of samples
        num_samples = convert_hex_to_decimal(sub_string[string_index:string_index + 4], "little")
        string_index += 4

        # Extract system timestamp
        sys_timestamp = convert_hex_to_decimal(sub_string[string_index:string_index + 8], "little")
        string_index += 8

        # Extract sequence number
        sequence_num = convert_hex_to_decimal(sub_string[string_index:string_index + 4], "little")
        string_index += 4

        if header != "WV_IMU":
            # Standard packet processing
            for _ in range(num_samples):
                current_sample = convert_hex_to_decimal(sub_string[string_index:string_index + sample_size * 2], "little")
                samples.append(current_sample)
                string_index += sample_size
        else:
            # Special handling for IMU packets
            for _ in range(num_samples):
                current_sample = []
                for subsample in range(6):
                    if subsample < 3:
                        current_subsample = convert_hex_to_decimal(sub_string[string_index:string_index + 4], "little")
                        string_index += 4
                        current_sample.append(current_subsample)
                    else:
                        gyro_index = string_index - 12 + (num_samples * 12)
                        current_subsample = convert_hex_to_decimal(sub_string[gyro_index:gyro_index + 4], "little")
                        gyro_index += 4
                        current_sample.append(current_subsample)
                samples.append(", ".join(map(str, current_sample)))

        # Store packet data
        packet_data = [header, RTC_timestamp, num_samples, sys_timestamp, sequence_num] + samples
        output_list.append(packet_data)

    # Find the maximum length of any list in output_list
    max_length = max(len(packet) for packet in output_list)

    # Pad all lists to the maximum length with None
    padded_list = [packet + [None] * (max_length - len(packet)) for packet in output_list]

    # Convert to DataFrame
    column_names = ["header", "RTC_timestamp", "num_samples", "sys_timestamp", "sequence_num"] + \
                   [f"sample_{i + 1}" for i in range(max_length - 5)]
    output_df = pd.DataFrame(padded_list, columns=column_names)

    return output_df

def extract_IMU_values(input_string, channel):
    """
    Extracts IMU values from the input string and returns the value of the specified channel.

    Args:
        input_string (str): A string containing IMU values separated by commas and spaces.
        channel (str): The channel to extract (e.g., 'accel_x', 'gyro_y').

    Returns:
        float: The extracted value for the specified channel.

    Raises:
        ValueError: If the specified channel is invalid.
    """
    # Split the input string by ", " to get numeric values
    values = [float(val) for val in input_string.split(", ")]

    # Assign the extracted values to respective variables
    accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = values

    # Return the requested channel value
    channel_map = {
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z,
    }

    if channel not in channel_map:
        raise ValueError("Invalid Channel")

    return channel_map[channel]

def process_data_frame_step_1(raw_df_output):
    # Step 1: Filter out WV_DIGITAL_EVENTS
    processed_df = raw_df_output[raw_df_output['header'] != "WV_DIGITAL_EVENTS"].copy()

    # Step 2: Turn sample columns into rows
    processed_df = processed_df.melt(
        id_vars=['header', 'RTC_timestamp', 'sys_timestamp', 'num_samples', 'sequence_num'],
        var_name='sample_num',
        value_name='value'
    )

    # Step 3: Filter out NAs
    processed_df = processed_df[~processed_df['value'].isna()]

    # Step 4: Turn sample_num into numeric
    processed_df['sample_num'] = processed_df['sample_num'].str.extract(r'(\d+)').astype(int)

    # Step 5: Create `elapsed_time` column
    processed_df['time_elapsed'] = np.where(
        processed_df['header'] == "WV_TRANST_IMPED",
        processed_df['sequence_num'] + (1 / 30) * processed_df['sample_num'],
        processed_df['sequence_num'] + (1 / 50) * processed_df['sample_num']
    )

    # Step 6: Rearrange columns
    processed_df = processed_df[['header', 'RTC_timestamp', 'time_elapsed', 'sequence_num', 'num_samples', 'value']]

    # Step 7: Sort by `time_elapsed`
    processed_df = processed_df.sort_values(by='time_elapsed')

    # Step 8: Convert `value` to string
    processed_df['value'] = processed_df['value'].astype(str)

    # Step 9: Create TTI column
    processed_df['TTI'] = np.where(
        processed_df['header'] == "WV_TRANST_IMPED",
        processed_df['value'],
        np.nan
    )

    # Step 10: Convert TTI to ohms
    processed_df['TTI'] = pd.to_numeric(processed_df['TTI'], errors='coerce')
    processed_df['TTI'] = (processed_df['TTI'] * 0.04176689) - 8.5538812

    return processed_df

def process_data_frame_step_2(processed_df):
    """
    Adds IMU acceleration and gyroscope data (accel_X, accel_Y, accel_Z, gyro_X, gyro_Y, gyro_Z)
    by applying `extract_IMU_values` function row-wise to the DataFrame.

    Args:
        processed_df (pd.DataFrame): DataFrame from the previous processing step.

    Returns:
        pd.DataFrame: Updated DataFrame with new columns for IMU values.
    """

    def extract_channel(header, value, channel):
        if header == "WV_TRANST_IMPED":
            return np.nan
        elif header == "WV_IMU":
            return extract_IMU_values(value, channel)
        else:
            return np.nan

    # Apply extract_channel function row-wise for each channel
    processed_df['accel_X'] = processed_df.apply(
        lambda row: extract_channel(row['header'], row['value'], "accel_x"), axis=1
    )
    processed_df['accel_Y'] = processed_df.apply(
        lambda row: extract_channel(row['header'], row['value'], "accel_y"), axis=1
    )
    processed_df['accel_Z'] = processed_df.apply(
        lambda row: extract_channel(row['header'], row['value'], "accel_z"), axis=1
    )
    processed_df['gyro_X'] = processed_df.apply(
        lambda row: extract_channel(row['header'], row['value'], "gyro_x"), axis=1
    )
    processed_df['gyro_Y'] = processed_df.apply(
        lambda row: extract_channel(row['header'], row['value'], "gyro_y"), axis=1
    )
    processed_df['gyro_Z'] = processed_df.apply(
        lambda row: extract_channel(row['header'], row['value'], "gyro_z"), axis=1
    )

    return processed_df

## Main program loop.

def main():
    while True:
        # Get the list of applicable files
        folder_path = "raw data"
        applicable_extensions = (".txt", ".hex")
        files = [f for f in os.listdir(folder_path) if f.endswith(applicable_extensions)]

        # Check if there are any applicable files in `raw data`
        if not files:
            print("No applicable files found in the `/raw data` folder. Program will terminate.")
            return  # Exit the program

        print("Below is a list of files that can be processed: \n")
        for idx, file_name in enumerate(files, start=1):
            print(f"[{idx}] {file_name}")
        print("\n")

        while True:
            try:
                choice = int(input("Enter the number of the file you want to process: "))
                if 1 <= choice <= len(files):
                    selected_file = files[choice - 1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(files)}.")
            except ValueError:
                print(f"Invalid input. Please enter a number between 1 and {len(files)}.")

        # Store the selected file name without extension
        selected_file_base = os.path.splitext(selected_file)[0]

        # Open the selected file
        with open(os.path.join(folder_path, selected_file), "r") as file:
            raw_hex = file.read()

        # Gets rid of the meaningless 0x prefix before every 2-digit hex value and removes commas from the string.
        # `hex_string` is now a clean hex string, ready to be processed.
        no_prefix = raw_hex.replace("0x", "")
        hex_string = "".join(no_prefix).replace(",", "")

        ## Existing logic for processing hex strings into clean DataFrame
        # Convert the clean hex string into packets and then to a DataFrame
        packet_list = parse_to_packets(hex_string)
        raw_df = parse_to_data_frame(packet_list)

        step_1_df = process_data_frame_step_1(raw_df)
        step_2_df = process_data_frame_step_2(step_1_df)
        clean_df = step_2_df.drop(columns=['value', 'header', 'RTC_timestamp', 'num_samples'])

        # Save the output
        os.makedirs("processed data", exist_ok=True)  # Ensure the folder exists
        output_file = os.path.join("processed data", f"{selected_file_base}_processed.csv")
        clean_df.to_csv(output_file, na_rep="NA", index=False)

        print(f"File processed successfully. The output file can be found in the `/processed data` folder.")

        # Ask user whether to process another file
        while True:
            restart = input("Would you like to process another file (y/n)? ").strip().lower()
            if restart == 'y':
                break  # Restart the script
            elif restart == 'n':
                print("Exiting program.")
                return  # Terminate the program
            else:
                print("Invalid input. Please respond with 'y' or 'n'.")

# Execute the program.
if __name__ == "__main__":
    main()