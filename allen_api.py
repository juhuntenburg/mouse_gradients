from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.core.structure_tree import StructureTree

oapi = OntologiesApi()
structure_graph = oapi.get_structures_with_sets([1])
structure_graph = StructureTree.clean_structures(structure_graph)

structure_graph = StructureTree.clean_structures(structure_graph)
tree = StructureTree(structure_graph)

name_map = tree.get_name_map()
cortexStructures = [name_map[i] for i in tree.descendant_ids([315])[0]]
