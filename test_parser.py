import sys
sys.path.insert(0, 'src')

from schematic_parser import SchematicParser

parser = SchematicParser()
result = parser.parse_file('fixed_all_files (1)/fixed_all_files/a_big_abandoned_house_out_of_stone_overgrown_with_spruce_trees_0001.schem')

if result is not None:
    print(f'SUCCESS! Parsed {len(result)} blocks')
    stats = parser.get_structure_stats(result)
    print(f'Solid blocks: {stats["solid_blocks"]}/{stats["total_blocks"]} ({stats["density"]:.1%})')
    print(f'Unique block types: {stats["unique_blocks"]}')
else:
    print('FAILED to parse file')
