## Packages.

## Packages.

from os import listdir as os_listdir, makedirs as os_makedirs, path as os_path
from progressbar import Bar as pgb_Bar, Percentage as pgb_Percentage, SimpleProgress as pgb_SimpleProgress, ProgressBar as pgb_ProgressBar
from pandas import DataFrame as pd_DataFrame, melt as pd_melt, to_numeric as pd_to_numeric
from numpy import nan as np_nan, where as np_where
from threading import Thread as td_Thread, Event as td_Event, Timer as td_Timer
from time import sleep as tme_sleep, time as tme_time
from itertools import cycle as iter_cycle
from sys import stdout as sys_stdout, exit as sys_exit

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

    print("[3/7]: Parsing hex string into packets... ")

    # Initialize the progress bar
    widgets = [
        pgb_Bar(marker='█', left='|', right='|', fill = '—'),  # Customized bar style
        ' ', pgb_Percentage(),  # Show percentage
        ' ', pgb_SimpleProgress(format='(%s)' % '%(value)d/%(max_value)d'),  # Show current/max
    ]
    bar = pgb_ProgressBar(widgets=widgets, max_value=hex_length)


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

        # Update the progress bar
        bar.update(index)

    bar.finish()

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
        pd_DataFrame: A DataFrame containing the decoded packet information.
    """
    print("[4/7]: Converting packets to dataframe...")

    # Initialize progress bar
    widgets = [
        pgb_Bar(marker='█', left='|', right='|', fill = '—'),  # Customized bar style
        ' ', pgb_Percentage(),  # Show percentage
        ' ', pgb_SimpleProgress(format='(%s)' % '%(value)d/%(max_value)d'),  # Show current/max
    ]
    bar = pgb_ProgressBar(widgets=widgets, max_value=len(hex_list))

    output_list = []

    # Iterate over the list of hex substrings
    for i, sub_string in enumerate(hex_list):
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

        # Update the progress bar
        bar.update(i + 1)

    bar.finish()

    # Find the maximum length of any list in output_list
    max_length = max(len(packet) for packet in output_list)

    # Pad all lists to the maximum length with None
    padded_list = [packet + [None] * (max_length - len(packet)) for packet in output_list]

    # Convert to DataFrame
    column_names = ["header", "RTC_timestamp", "num_samples", "sys_timestamp", "sequence_num"] + \
                   [f"sample_{i + 1}" for i in range(max_length - 5)]
    output_df = pd_DataFrame(padded_list, columns=column_names)

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
    """
    Processes the raw DataFrame step by step.
    Includes a charging-style progress bar indicating progress through 10 steps.

    Args:
        raw_df_output (pd_DataFrame): The raw input DataFrame.

    Returns:
        pd_DataFrame: The processed DataFrame.
    """
    print("[5/7]: Processing dataframe...")

    # Initialize the progress bar
    widgets = [
        pgb_Bar(marker='█', left='|', right='|', fill = '—'),  # Charging bar style
        ' ', pgb_Percentage(),  # Show percentage
        ' ', pgb_SimpleProgress(format='(%s)' % '%(value)d/%(max_value)d'),  # Show current/max
    ]
    bar = pgb_ProgressBar(widgets=widgets, max_value=10)

    # Step 1: Filter out WV_DIGITAL_EVENTS
    processed_df = raw_df_output[raw_df_output['header'] != "WV_DIGITAL_EVENTS"].copy()
    bar.update(1)

    # Step 2: Turn sample columns into rows
    processed_df = processed_df.melt(
        id_vars=['header', 'RTC_timestamp', 'sys_timestamp', 'num_samples', 'sequence_num'],
        var_name='sample_num',
        value_name='value'
    )
    bar.update(2)

    # Step 3: Filter out NAs
    processed_df = processed_df[~processed_df['value'].isna()]
    bar.update(3)

    # Step 4: Turn sample_num into numeric
    processed_df['sample_num'] = processed_df['sample_num'].str.extract(r'(\d+)').astype(int)
    bar.update(4)

    # Step 5: Create `elapsed_time` column
    processed_df['time_elapsed'] = np_where(
        processed_df['header'] == "WV_TRANST_IMPED",
        processed_df['sequence_num'] + (1 / 30) * processed_df['sample_num'],
        processed_df['sequence_num'] + (1 / 50) * processed_df['sample_num']
    )
    bar.update(5)

    # Step 6: Rearrange columns
    processed_df = processed_df[['header', 'RTC_timestamp', 'time_elapsed', 'sequence_num', 'num_samples', 'value']]
    bar.update(6)

    # Step 7: Sort by `time_elapsed`
    processed_df = processed_df.sort_values(by='time_elapsed')
    bar.update(7)

    # Step 8: Convert `value` to string
    processed_df['value'] = processed_df['value'].astype(str)
    bar.update(8)

    # Step 9: Create TTI column
    processed_df['TTI'] = np_where(
        processed_df['header'] == "WV_TRANST_IMPED",
        processed_df['value'],
        np_nan
    )
    bar.update(9)

    # Step 10: Convert TTI to ohms
    processed_df['TTI'] = pd_to_numeric(processed_df['TTI'], errors='coerce')
    processed_df['TTI'] = (processed_df['TTI'] * 0.04176689) - 8.5538812
    bar.update(10)

    bar.finish()

    return processed_df

def process_data_frame_step_2(processed_df):
    """
    Adds IMU acceleration and gyroscope data (accel_X, accel_Y, accel_Z, gyro_X, gyro_Y, gyro_Z)
    by applying `extract_IMU_values` function row-wise to the DataFrame.

    Includes a charging-style progress bar for row processing.

    Args:
        processed_df (pd_DataFrame): DataFrame from the previous processing step.

    Returns:
        pd_DataFrame: Updated DataFrame with new columns for IMU values.
    """

    def extract_channel(header, value, channel):
        if header == "WV_TRANST_IMPED":
            return np_nan
        elif header == "WV_IMU":
            return extract_IMU_values(value, channel)
        else:
            return np_nan

    print("[6/7]: Extracting IMU values...")

    # Calculate the total number of rows at the start
    num_rows = len(processed_df)
    if num_rows == 0:
        print("The DataFrame is empty. Skipping Step 2.")
        return processed_df  # Return empty DataFrame

    # Initialize the progress bar
    widgets = [
        pgb_Bar(marker='█', left='|', right='|', fill = '—'),  # Charging bar style
        ' ', pgb_Percentage(),  # Show percentage
        ' ', pgb_SimpleProgress(format='(%s)' % '%(value)d/%(max_value)d'),  # Show current/max
    ]
    bar = pgb_ProgressBar(widgets=widgets, max_value=num_rows)

    # Initialize empty columns
    processed_df['accel_X'] = np_nan
    processed_df['accel_Y'] = np_nan
    processed_df['accel_Z'] = np_nan
    processed_df['gyro_X'] = np_nan
    processed_df['gyro_Y'] = np_nan
    processed_df['gyro_Z'] = np_nan

    # Iterate through each row and update progress bar
    for idx, (index, row) in enumerate(processed_df.iterrows()):
        header = row['header']
        value = row['value']

        # Update IMU columns for the current row
        processed_df.at[index, 'accel_X'] = extract_channel(header, value, "accel_x")
        processed_df.at[index, 'accel_Y'] = extract_channel(header, value, "accel_y")
        processed_df.at[index, 'accel_Z'] = extract_channel(header, value, "accel_z")
        processed_df.at[index, 'gyro_X'] = extract_channel(header, value, "gyro_x")
        processed_df.at[index, 'gyro_Y'] = extract_channel(header, value, "gyro_y")
        processed_df.at[index, 'gyro_Z'] = extract_channel(header, value, "gyro_z")

        # Update the progress bar
        bar.update(idx + 1)

    bar.finish()
    return processed_df

def save_with_progress_bar(clean_df, output_file):
    """
    Save the DataFrame to a file with a progress bar indicating progress.

    Args:
        clean_df (pd_DataFrame): The DataFrame to save.
        output_file (str): The path to save the file.
    """
    total_rows = len(clean_df)

    print("[7/7]: Writing output csv file...")

    # Initialize progress bar
    widgets = [
        pgb_Bar(marker='█', left='|', right='|', fill = '—'),  # Charging bar style
        ' ', pgb_Percentage(),  # Show percentage
        ' ', pgb_SimpleProgress(format='(%s)' % '%(value)d/%(max_value)d'),  # Show current/max
    ]
    progress = pgb_ProgressBar(widgets=widgets, maxval=total_rows).start()

    # Write the file incrementally
    with open(output_file, 'w') as file:
        # Write the header
        file.write(",".join(clean_df.columns) + "\n")
        progress.update(0)

        # Write rows incrementally
        for i, row in enumerate(clean_df.itertuples(index=False, name=None)):
            file.write(",".join(map(str, row)) + "\n")
            progress.update(i + 1)

    progress.finish()

def prompt_with_timeout(prompt_message, timeout):
    """
    Prompt the user with a message and terminate if no response is given within the timeout.

    Args:
        prompt_message (str): The message to display for the prompt.
        timeout (int): The timeout duration in seconds.

    Returns:
        str: The user's input, or None if the timeout expires.
    """
    def timeout_handler():
        """Exit the program if the timeout expires."""
        print("\nNo response given. Exiting program.")
        sys_exit()

    # Set up a timer to terminate the program after the timeout
    timer = td_Timer(timeout, timeout_handler)
    timer.start()

    try:
        # Prompt the user for input
        response = input(prompt_message).strip()
    finally:
        # Cancel the timer if the user responds
        timer.cancel()

    return response

def read_file_with_progress(folder_path, selected_file, chunk_size=1024):
    file_path = os_path.join(folder_path, selected_file)
    file_size = os_path.getsize(file_path)  # Get the total file size

    print("[1/7]: Reading raw input file...")
    # Initialize progress bar.
    widgets = [
        pgb_Bar(marker='█', left='|', right='|', fill = '—'),  # Customized bar style
        ' ', pgb_Percentage(),  # Show percentage
        ' ', pgb_SimpleProgress(format='(%s)' % '%(value)d/%(max_value)d'),  # Show current/max
    ]
    progress = pgb_ProgressBar(widgets=widgets, maxval=file_size).start()

    raw_hex = []  # To store file content
    with open(file_path, "r") as file:
        read_bytes = 0
        while chunk := file.read(chunk_size):
            raw_hex.append(chunk)
            read_bytes += len(chunk)
            progress.update(read_bytes)  # Update the progress bar

    progress.finish()
    return "".join(raw_hex)  # Combine all chunks into a single string

def clean_hex_string(raw_hex):
    """
    Cleans the raw_hex string by removing "0x" prefixes and commas, with a single progress bar.

    Args:
        raw_hex (str): The raw hex string to clean.

    Returns:
        str: A cleaned hex string.
    """
    total_size = len(raw_hex) + len(raw_hex.replace("0x", ""))  # Total progress for both operations


    print("[2/7]: Preprocessing...")
    # Initialize progress bar.
    widgets = [
        pgb_Bar(marker='█', left='|', right='|', fill = '—'),  # Customized bar style
        ' ', pgb_Percentage(),  # Show percentage
        ' ', pgb_SimpleProgress(format='(%s)' % '%(value)d/%(max_value)d'),  # Show current/max
    ]
    progress = pgb_ProgressBar(widgets=widgets, maxval=total_size).start()

    # Step 1: Remove "0x"
    no_prefix = raw_hex.replace("0x", "")
    for i in range(len(raw_hex)):  # Simulate progress for this step
        progress.update(i + 1)

    # Step 2: Remove commas
    hex_string = no_prefix.replace(",", "")
    for i in range(len(no_prefix)):  # Simulate progress for this step
        progress.update(len(raw_hex) + i + 1)

    progress.finish()

    # Debugging: Verify final result
    assert hex_string == raw_hex.replace("0x", "").replace(",", ""), "Final result mismatch"

    return hex_string


## Main program loop.

def main():
    while True:
        # Get the list of applicable files
        folder_path = "raw data"
        applicable_extensions = (".txt", ".hex")
        files = [f for f in os_listdir(folder_path) if f.endswith(applicable_extensions)]

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

        print("\n")

        # Store the selected file name without extension
        selected_file_base = os_path.splitext(selected_file)[0]

        # Record the start time
        start_time = tme_time()  

        # Open the selected file
        raw_hex = read_file_with_progress(folder_path, selected_file)

        # Gets rid of the meaningless 0x prefix before every 2-digit hex value and removes commas from the string.
        # `hex_string` is now a clean hex string, ready to be processed.
        hex_string = clean_hex_string(raw_hex)

        ## Existing logic for processing hex strings into clean DataFrame
        # Convert the clean hex string into packets and then to a DataFrame
        packet_list = parse_to_packets(hex_string)
        raw_df = parse_to_data_frame(packet_list)

        step_1_df = process_data_frame_step_1(raw_df)
        step_2_df = process_data_frame_step_2(step_1_df)
        clean_df = step_2_df.drop(columns=['value', 'header', 'RTC_timestamp', 'num_samples'])

        # Save the output
        os_makedirs("processed data", exist_ok=True)  # Ensure the folder exists
        output_file = os_path.join("processed data", f"{selected_file_base}_processed.csv")
        save_with_progress_bar(clean_df, output_file)

        elapsed_time = tme_time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)

        print("\n")

        if minutes < 1:
            print(f"File processed successfully in {seconds:.2f} seconds.")
        else:
            print(f"File processed successfully in {int(minutes)} minutes and {seconds:.2f} seconds.")

        print("The output file can be found in the `/processed data` folder. \n")

        # Ask user whether to process another file wieth a 300-second timeout
        while True:
            restart = prompt_with_timeout("Would you like to process another file (y/n)? ", 300).lower()
            if restart == 'y':
                break  # Restart the script
            elif restart == 'n':
                print("Exiting program.")
                sys_exit()  # Terminate the program
            else:
                print("Invalid input. Please respond with 'y' or 'n'.") 

# Execute the program.
if __name__ == "__main__":
    main()