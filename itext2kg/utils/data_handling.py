from typing import Literal, List, Tuple

class DataHandler:
    """
    A class to handle and manipulate data, particularly focusing on processing,
    deduplicating, and managing relationships and entities in data records.
    """
    
    def __init__(self):
        """Initialize the DataHandler instance."""
        pass

    def process(self, data: dict, data_type: Literal['entity', 'relation']) -> dict:
        """
        Process the given data based on its type (entity or relation) by normalizing and cleaning it.
        
        Args:
        data (dict): The data to process.
        data_type (Literal['entity', 'relation']): The type of the data to process.
        
        Returns:
        dict: The processed data.
        """
        data = data.copy()
        if data_type == 'relation':
            # Normalize the start and end nodes by converting them to lowercase.
            data["startNode"] = data["startNode"].lower()
            data["endNode"] = data["endNode"].lower()
            # Replace spaces, dashes, periods, and '&' in names with underscores or 'and'.
            data["name"] = data["name"].replace(" ", "_").replace("-", "_").replace(".", "_").replace("&", "and")
        elif data_type == 'entity':
            # Replace spaces, dashes, periods, and '&' in labels with underscores or 'and'.
            data["label"] = data["label"].replace(" ", "_").replace("-", "_").replace(".", "_").replace("&", "and")
        
        return data

    def remove_duplicates(self, records: List[dict], data_type: Literal['entity', 'relation']) -> List[dict]:
        """
        Remove duplicate records from the data based on their unique identifiers.
        
        Args:
        records (List[dict]): A list of records to deduplicate.
        data_type (Literal['entity', 'relation']): The type of the data to deduplicate.
        
        Returns:
        List[dict]: A list of unique records.
        """
        seen = set()
        unique_records = []
        for record in records:
            # Create an identifier tuple based on the type of the data.
            if data_type == 'relation':
                identifier = (record['startNode'], record['endNode'], record['name'])
            elif data_type == 'entity':
                identifier = record['name']
            
            # Add to unique records if identifier is not seen before.
            if identifier not in seen:
                seen.add(identifier)
                unique_records.append(record)
        
        return unique_records

    def handle_data(self, data: List[dict], data_type: Literal['entity', 'relation']) -> List[dict]:
        """
        Process and remove duplicates from a list of data records.
        
        Args:
        data (List[dict]): The data to handle.
        data_type (Literal['entity', 'relation']): The type of the data to handle.
        
        Returns:
        List[dict]: Processed and unique data records.
        """
        processed_data = [self.process(item, data_type=data_type) for item in data]
        unique_data = self.remove_duplicates(records=processed_data, data_type=data_type)
        return unique_data
    
    def find_relations_with_isolated_entities(self, global_entities: List[dict], relations: List[dict]) -> List[dict]:
        """
        Identify relations that reference entities not listed in the provided global entities.
        
        Args:
        global_entities (List[dict]): A list of global entities with their details.
        relations (List[dict]): A list of relations to check for isolated entities.
        
        Returns:
        List[dict]: A list of relations involving entities not found in the global entities list.
        """
        isolated_entities = relations.copy()
        global_entities_names = [ent["name"] for ent in global_entities]
        for rel in relations:
            if rel["startNode"] in global_entities_names and rel["endNode"] in global_entities_names:
                isolated_entities.remove(rel)
        return isolated_entities

    def match_relations_with_isolated_entities(self, global_entities: List[dict], relations: List[dict], matcher, embedding_calculator) -> List[dict]:
        """
        Match and replace the names of isolated entities in relations with the closest matches from global entities,
        using an embedding calculator and a matching function.
        
        Args:
        global_entities (List[dict]): A list of global entities.
        relations (List[dict]): A list of relations that may include isolated entities.
        matcher (Callable): A function to find the closest match for an entity based on embeddings.
        embedding_calculator (Callable): A function to calculate embeddings for entities.
        
        Returns:
        List[dict]: Updated list of relations with matched entity names.
        """
        isolated_entities = relations.copy()
        global_entities_names = [ent["name"] for ent in global_entities]

        for rel in relations:
            if rel["startNode"] not in global_entities_names:
                # Calculate embeddings for the isolated start node and find the closest match.
                node = {"name": rel["startNode"], "label": "entity", "properties": {"embeddings": embedding_calculator(rel["startNode"])}}
                closest_node = matcher(node)
                rel["startNode"] = closest_node["name"]

            if rel["endNode"] not in global_entities_names:
                # Repeat the matching process for the isolated end node.
                node = {"name": rel["endNode"], "label": "entity", "properties": {"embeddings": embedding_calculator(rel["endNode"])}}
                closest_node = matcher(node)
                rel["endNode"] = closest_node["name"]

        return isolated_entities

    def find_isolated_entities(self, global_entities: List[dict], relations: List[dict]) -> List[dict]:
        """
        Identify entities from the global list that are not referenced in any relation.
        
        Args:
        global_entities (List[dict]): A list of global entities.
        relations (List[dict]): A list of relations to check for entity references.
        
        Returns:
        List[dict]: A list of entities that are not involved in any relations.
        """
        relation_nodes = set(rel["startNode"] for rel in relations) | set(rel["endNode"] for rel in relations)
        isolated_entities = [ent for ent in global_entities if ent["name"] not in relation_nodes]
        return isolated_entities