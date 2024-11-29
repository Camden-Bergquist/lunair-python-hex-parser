# Python-Based Hex Parser

Data parsing executable written for Lunair Medical in Python. Converts raw hex data from an IPG into a readable csv file in format reminiscent of the Integer Programmer's marker mode output.


## Components:

The compiled program comes in the form of one executable file (`Hex Parser (vX.X.X).exe`), three folders, and this readme. Two of the folders — `/raw_data` and `/processed_data` — are relevant to the operation of the executable. The other, `/<placeholder>`, contains resources necessary for the program to run, and should be left alone.

Four files are contained in the `/raw_data` folder: `example_data.txt`, `example_data.hex`, `example_data_2_(large).txt`, and `example_data_2_(large).hex`. There is no functional difference between the `.hex` and `.txt` extensions for files of the same name; they are present in order to communicate that the parser can work with either file type. In effect, this means that there are really only two sets of sample data.

The `\processed_data` folder will be empty upon installation.


## Use:

To use the parser, open the `Hex Parser (vX.X.X).exe` file. The terminal will read the contents of the `/raw_data` folder, and then prompt the user to select which of the files within, as denoted by assigned numbers, to parse. This list will contain solely files with the `.hex` and `.txt` file extensions, and ignore everything else. **In order for the program to properly parse a file, that file must be contained within the `/raw_data` folder before the executable is run, in addition to being the proper file type (`.hex`, `.txt`).** If no applicable files are present in the `raw_data` folder, the program will terminate.

Once the user selects an applicable file from the list, the parser wil read, process, and write the file in a series of steps. Loading bars for each step are present to ensure that the program hasn't stalled in some way. Large files are liable to take several minutes to process. Processed files can be found in the `/processed_data` folder, and are always formatted as csv files.

When the parser is finished processing and writing a file, it will ask the user whether they wish to parse another file (by typing y), or exit the program (by typing n). The parser will also terminate if, after it finishes processing a file, it recieves no input for a period of five minutes. Termination in this manner poses no risk to output files, as they will be necessarily already finished.
