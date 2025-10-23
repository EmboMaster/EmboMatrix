#python social_gen_test.py
import social_character_gen
import read_files

social_character_gen.social_character_generation_in_detail(read_files.read_lines_to_array("extracted_rooms.txt")[42])