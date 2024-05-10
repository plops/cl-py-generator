#!/bin/bash

# Array containing video IDs and titles
video_data=(
    "VMj-3S1tku0 The_spelled-out_intro_to_neural_networks_and_backpropagation"
    "PaCmpygFfXo The_spelled-out_intro_to_language_modeling_building_makemore"
    "TCH_1BHY58I Makemore_Part_2_MLP"
    "P6sfmUTpUmc Makemore_Part_3_Activations_BatchNorm" 
    "q8SA3rM6ckI Makemore_Part_4_Backprop"
    "t3YJ5hKiMQ0 Makemore_Part_5_WaveNet"
    "kCc8FmEb1nY GPT_from_scratch" 
    "zduSFxRajkE GPT_Tokenizer"
)

# Loop through each video
for video in "${video_data[@]}"; do
    # Extract video ID and title
    video_id=$(echo "$video" | cut -d' ' -f1)
    video_title=$(echo "$video" | cut -d' ' -f2-)

    # Run the script with video ID and title as arguments
    ./transcript.sh "$video_id" "$video_title"
done

    
