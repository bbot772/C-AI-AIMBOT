#ifndef TYPES_H
#define TYPES_H

#include <string>

// Simple 2D Vector
struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

// A single bounding box detection
struct Detection
{
    // Bounding box coordinates
    float xmin, ymin, xmax, ymax;

    // The center of the bounding box
    // This is a derived value, not directly from the model
    int x_center;
    int y_center;

    // The width and height of the bounding box
    // Also a derived value
    int w;
    int h;
    
    // Confidence score for the detection
    float confidence;

    // ID and name of the detected class
    int class_id;
    std::string class_name;
};


#endif // TYPES_H 