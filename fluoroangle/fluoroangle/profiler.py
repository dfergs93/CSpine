from fluoroangle import process_patient_images
import cProfile

folder_path = '/Users/duncanferguson/Desktop/CSpine/dicom_data/1.2.826.0.1.3680043.1641'

# Call the function with cProfile
profiler = cProfile.Profile()
profiler.enable()

# Call the function you want to profile
axial_stack = process_patient_images(folder_path)

profiler.disable()
profiler.print_stats(sort='time')
