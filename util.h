#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using std::vector;
using std::string;


string ReadFile(string file_name) {
    // Create a string to hold the data
    string data;
    // Create a string to hold each line of the file
    string line;
    // Create an input file stream
    std::ifstream file;
    // Open the file
    file.open(file_name);
    // Check if the file is open
    if (file.is_open()) {
        // While there is still data to read
        while (getline(file, line)) {
            // Add the line to the data string
            data += line;
            // Add a newline character
            data += "\n";
        }
        // Close the file
        file.close();
    }
    // Return the data
    return data;
}


vector<vector<float>> ReadCsvMatrix(string csv_string) {
    // Create a vector to hold the matrix
    vector<vector<float>> matrix;
    // Create a vector to hold the current row
    vector<float> row;
    // Create a string to hold the current value
    string value;
    // Loop through the characters in the string
    for (int i = 0; i < csv_string.size(); i++) {
        // If the current character is a comma
        if (csv_string[i] == ',') {
            // Add the current value to the row
            row.push_back(std::stof(value));
            // Clear the value
            value = "";
        }
        // If the current character is a newline
        else if (csv_string[i] == '\n') {
            // Add the current value to the row
            row.push_back(std::stoi(value));
            // Clear the value
            value = "";
            // Add the current row to the matrix
            matrix.push_back(row);
            // Clear the row
            row.clear();
        }
        // If the current character is not a comma or newline
        else {
            // Add the current character to the value
            value += csv_string[i];
        }
    }
    // Return the matrix
    return matrix;
}
