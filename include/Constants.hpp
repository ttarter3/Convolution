#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <iostream>
#include <string>
#include <unordered_map>

class Constants {
private:
    const double c = 299792458.0; // Speed of light in meters per second
    const double pi = 3.141592653589793238462643;

    // Conversion factors for different units
    std::unordered_map<std::string, double> conversionFactors = {
        {"m/s", 1.0},
        {"km/h", 3.6},
        {"mi/s", 0.000621371},
        {"c", 1.0} // Speed of light in its own units
    };

public:
    double C(const std::string& unit = "m/s") const {
        // Check if the unit exists in conversionFactors
        if (conversionFactors.find(unit) != conversionFactors.end()) {
            return c / conversionFactors.at(unit);
        } else {
            std::cerr << "Error: Unsupported unit \"" << unit << "\"" << std::endl;
            return -1.0; // Return a negative value to indicate error
        }
    }

    double PI() const {
        return pi;
    }
};

#endif // CONSTANTS_HPP