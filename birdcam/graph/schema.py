from birdcam.graph.cypher import SCHEMA_STATEMENTS

def init_schema(driver):
    for q in SCHEMA_STATEMENTS:
        driver.execute(q)
