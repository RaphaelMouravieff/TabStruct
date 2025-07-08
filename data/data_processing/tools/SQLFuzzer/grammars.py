from typing import Dict, Union, Any, Tuple, List
import pandas as pd
import random
# defining custom type
Option = Dict[str, Any]
Expansion = Union[str, Tuple[str, Option]]
Grammar = Dict[str, List[Expansion]]



def dummy_grammar() -> Grammar :
    """
    Dummy grammar to initialize the Fuzzer. Not meant to be used in training time
    :return: grammar
    """

    SQL_GRAMMAR: Grammar = {
        '<Query>': ['OK'],
    }

    return SQL_GRAMMAR







def sql_grammar_SUM(table : pd.DataFrame) -> Grammar :
    """
    Grammar generating sql query of the form : SELECT <field> from w, The input table is ignored.
    :param table: ignored
    :return: grammar
    e.g., SELECT sum ( c1 ) FROM w
    """
    db = "w"

    fields = table.columns.tolist()

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            ['select SUM(<Attribute>) from <Relation>'],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    return SQL_GRAMMAR


def sql_grammar_GB(table : pd.DataFrame) -> Grammar :

    db = "w"
    # Assuming the DataFrame's columns are to be used as potential fields
    fields = table.columns.tolist()

    SQL_GRAMMAR: Grammar = {
        '<Query>': ['SELECT <Attribute> FROM <Relation> GROUP BY <GroupByAttribute>'],
        
        '<Attribute>': fields,

        '<GroupByAttribute>': fields,

        '<Relation>': [db],
    }


    return SQL_GRAMMAR

def sql_grammar_OB(table : pd.DataFrame) -> Grammar :

    db = "w"
    # Assuming the DataFrame's columns are to be used as potential fields
    fields = table.columns.tolist()

    SQL_GRAMMAR: Grammar = {
        '<Query>': ['SELECT <Attribute> FROM <Relation> ORDER BY <GroupByAttribute>'],
        
        '<Attribute>': fields,

        '<GroupByAttribute>': fields,

        '<Relation>': [db],
    }


    return SQL_GRAMMAR

def sql_grammar_WHERE_composition(table : pd.DataFrame):
    db = "w"

    fields = [f"c{i}" for i in range(5) ]
    comparators = ["<Comparator>"]
    types = ["<int>"]

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            ['select <SelList> from <FromList> where <Condition>'],


        '<SelList>':
            ['<Attribute>', '<SelList>, <Attribute>'],

        '<FromList>':
            ['<Relation>'],

        '<Condition>':
            ['<Comparison>', '<Condition> AND <Comparison>',
                '<Condition> OR <Comparison>'],

        '<Comparison>':
            [f'{f} {c} {t}' for f, c, t in zip(fields, comparators, types)],

        '<Comparator>': ['<', '<=',  '<LAngle><RAngle>', '>=', '>'],

        '<LAngle>': ['<'],

        '<RAngle>': ['>'],

        '<Relation>': [db],

        '<Attribute>': fields,

        # comparators:
        '<int>': [f'{i}' for i in range(999)],

    }

    return SQL_GRAMMAR

def sql_grammar_WHEREOR1(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form: SELECT <field> FROM w WHERE <field1> = <field_value1> AND <field2> = <field_value2>.
    The output query is guaranteed to work on the provided table.
    :param table: table from which the field and field values are extracted
    :return: grammar
    e.g., SELECT c1 FROM w WHERE c2 = v2 AND c3 = v3
    """

    # db name
    db = "w"

    # extract columns names
    fields = list(table.columns)

    # extract values per column
    value_per_column = {
        f"<{k}_values>": [str(x) for x in table[k].tolist() if pd.notnull(x)] for k in fields
    }

    SQL_GRAMMAR = {
        '<Query>':
            ['SELECT <SelList> FROM <FromList> WHERE <Condition>'],

        '<SelList>':
            ['<Attribute>'],

        '<FromList>':
            ['<Relation>'],

        '<Condition>':
            ['<Comparison1> OR <Comparison2>'],

        '<Comparison1>':
            [f'{field} = <{field}_values>' for field in fields],

        '<Comparison2>':
            [f'{field} = <{field}_values>' for field in fields],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    SQL_GRAMMAR.update(value_per_column)

    return SQL_GRAMMAR


def sql_grammar_WHEREOR2(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form: SELECT <field> FROM w WHERE <field1> = <field_value1> OR <field2> = <field_value2> OR <field3> = <field_value3>.
    The output query is guaranteed to work on the provided table.
    :param table: table from which the field and field values are extracted
    :return: grammar
    e.g., SELECT c1 FROM w WHERE c2 = v2 OR c3 = v3 OR c4 = v4
    """

    # db name
    db = "w"

    # extract columns names
    fields = list(table.columns)

    # extract values per column
    value_per_column = {
        f"<{k}_values>": [str(x) for x in table[k].tolist() if pd.notnull(x)] for k in fields
    }

    SQL_GRAMMAR = {
        '<Query>':
            ['SELECT <SelList> FROM <FromList> WHERE <Condition>'],

        '<SelList>':
            ['<Attribute>'],

        '<FromList>':
            ['<Relation>'],

        '<Condition>':
            ['<Comparison1> OR <Comparison2> OR <Comparison3>'],

        '<Comparison1>':
            [f'{field} = <{field}_values>' for field in fields],

        '<Comparison2>':
            [f'{field} = <{field}_values>' for field in fields],

        '<Comparison3>':
            [f'{field} = <{field}_values>' for field in fields],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    SQL_GRAMMAR.update(value_per_column)

    return SQL_GRAMMAR

def sql_grammar_WHEREOR3(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form: SELECT <field> FROM w WHERE <field1> = <field_value1> OR <field2> = <field_value2> OR <field3> = <field_value3> OR <field4> = <field_value4>.
    The output query is guaranteed to work on the provided table.
    :param table: table from which the field and field values are extracted
    :return: grammar
    e.g., SELECT c1 FROM w WHERE c2 = v2 OR c3 = v3 OR c4 = v4 OR c5 = v5
    """

    # db name
    db = "w"

    # extract columns names
    fields = list(table.columns)

    # extract values per column
    value_per_column = {
        f"<{k}_values>": [str(x) for x in table[k].tolist() if pd.notnull(x)] for k in fields
    }

    SQL_GRAMMAR = {
        '<Query>':
            ['SELECT <SelList> FROM <FromList> WHERE <Condition>'],

        '<SelList>':
            ['<Attribute>'],

        '<FromList>':
            ['<Relation>'],

        '<Condition>':
            ['<Comparison1> OR <Comparison2> OR <Comparison3> OR <Comparison4>'],

        '<Comparison1>':
            [f'{field} = <{field}_values>' for field in fields],

        '<Comparison2>':
            [f'{field} = <{field}_values>' for field in fields],

        '<Comparison3>':
            [f'{field} = <{field}_values>' for field in fields],

        '<Comparison4>':
            [f'{field} = <{field}_values>' for field in fields],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    SQL_GRAMMAR.update(value_per_column)

    return SQL_GRAMMAR

def sql_grammar_L1(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form: SELECT <field> FROM <table> LIMIT 1.
    The output query is guaranteed to work on the provided table.
    :param table: table from which the field is extracted
    :return: grammar
    e.g., SELECT c1 FROM w LIMIT 1
    """
    
    # db name
    db = "w"
    
    # extract column names
    fields = table.columns.tolist()

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            ['SELECT <Attribute> FROM <Relation> LIMIT 1'],

        '<Attribute>': fields,

        '<Relation>': [db],
    }

    return SQL_GRAMMAR


def sql_grammar_L2(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form: SELECT <field> FROM <table> LIMIT 2.
    The output query is guaranteed to work on the provided table.
    :param table: table from which the field is extracted
    :return: grammar
    e.g., SELECT c1 FROM w LIMIT 2
    """
    
    # db name
    db = "w"
    
    # extract column names
    fields = table.columns.tolist()

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            ['SELECT <Attribute> FROM <Relation> LIMIT 2'],

        '<Attribute>': fields,

        '<Relation>': [db],
    }

    return SQL_GRAMMAR

def sql_grammar_L3(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form: SELECT <field> FROM <table> LIMIT 3.
    The output query is guaranteed to work on the provided table.
    :param table: table from which the field is extracted
    :return: grammar
    e.g., SELECT c1 FROM w LIMIT 3
    """
    
    # db name
    db = "w"
    
    # extract column names
    fields = table.columns.tolist()

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            ['SELECT <Attribute> FROM <Relation> LIMIT 3'],

        '<Attribute>': fields,

        '<Relation>': [db],
    }

    return SQL_GRAMMAR



def sql_grammar_LIMIT(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form: SELECT <field> FROM <table> LIMIT N, 
    where N is randomly chosen between 1, 2, or 3.
    The output query is guaranteed to work on the provided table.
    :param table: table from which the field is extracted
    :return: grammar
    e.g., SELECT c1 FROM w LIMIT N
    """

    # db name
    db = "w"

    # extract column names
    fields = table.columns.tolist()

    # randomly choose the LIMIT value between 1, 2, or 3
    limit_value = random.choice([1, 2, 3])

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            [f'SELECT <Attribute> FROM <Relation> LIMIT {limit_value}'],

        '<Attribute>': fields,

        '<Relation>': [db],
    }

    return SQL_GRAMMAR


def sql_grammar_SELECT(table : pd.DataFrame) -> Grammar :
    """
    Grammar generating sql query of the form : SELECT <field> from w, The input table is ignored.
    :param table: ignored
    :return: grammar
    e.g., SELECT c1 FROM w
    """
    db = "w"

    fields = table.columns.tolist()

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            ['select <Attribute> from <Relation>'],

        '<Attribute>': fields,

        '<Relation>': [db],
    }

    return SQL_GRAMMAR

def sql_grammar_WHERE(table : pd.DataFrame) -> Grammar :
    """
    Grammar generating sql query of the form : SELECT <field> from w WHERE <field> = <field_value>.
    The output query is guaranty to be working on the provided table
    :param table: table from which the field and field values are extracted
    :return: grammar
    e.g., SELECT c1 FROM w WHERE c2 = v2
    """

    # db name
    db = "w"

    # extract columns names
    fields = list(table.columns)

    # extract values per column
    value_per_column = {
        f"<{k}_values>" : [str(x) for x in table[k].tolist() if x!=""] or ["0"] for k in fields
    }

    SQL_GRAMMAR = {
        '<Query>':
            ['select <SelList> from <FromList> where <Condition>'],

        '<SelList>':
            ['<Attribute>'],

        '<FromList>':
            ['<Relation>'],

        '<Condition>':
            ['<Comparison>'],

        '<Comparison>':
            [f'{field} = <{field}_values>' for field in fields],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    SQL_GRAMMAR.update(value_per_column)

    return SQL_GRAMMAR



def sql_grammar_NOTWHERE(table : pd.DataFrame) -> Grammar :
    """
    Grammar generating sql query of the form : SELECT <field> from w WHERE <field> = <field_value>.
    The output query is guaranty to be working on the provided table
    :param table: table from which the field and field values are extracted
    :return: grammar
    e.g., SELECT c1 FROM w WHERE c2 = v2
    """

    # db name
    db = "w"

    # extract columns names
    fields = list(table.columns)

    # extract values per column
    value_per_column = {
        f"<{k}_values>" : [str(x) for x in table[k].tolist() if x!=""] or ["0"] for k in fields
    }

    SQL_GRAMMAR = {
        '<Query>':
            ['select <SelList> from <FromList> where <Condition>'],

        '<SelList>':
            ['<Attribute>'],

        '<FromList>':
            ['<Relation>'],

        '<Condition>':
            ['<Comparison>'],

        '<Comparison>':
            [f'{field} != <{field}_values>' for field in fields],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    SQL_GRAMMAR.update(value_per_column)

    return SQL_GRAMMAR

def sql_grammar_AND(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form: SELECT <field> FROM w WHERE <field1> = <value1> AND <field2> = <value2>.
    The output query is guaranteed to work on the provided table.
    :param table: table from which the fields and field values are extracted
    :return: grammar
    e.g., SELECT c1 FROM w WHERE c2 = v2 AND c3 = v3
    """
    
    # db name
    db = "w"
    
    # extract column names
    fields = list(table.columns)
    
    # extract values per column
    value_per_column = {
        f"<{k}_values>": [str(x) for x in table[k].tolist() if pd.notnull(x)] or ["0"] for k in fields
    }

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            ['SELECT <SelList> FROM <FromList> WHERE <Condition>'],

        '<SelList>':
            ['<Attribute>'],

        '<FromList>':
            ['<Relation>'],

        '<Condition>':
            ['<Comparison1> AND <Comparison2>'],

        '<Comparison1>':
            [f'{field} = <{field}_values>' for field in fields],

        '<Comparison2>':
            [f'{field} = <{field}_values>' for field in fields],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    SQL_GRAMMAR.update(value_per_column)

    return SQL_GRAMMAR

def sql_grammar_nested_select(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form:
    SELECT <field1> FROM <table> WHERE <field2> = (SELECT <field3> FROM <table> WHERE <field4> = <value>) + 1.
    The output query is guaranteed to work on the provided table.
    :param table: table from which the fields and field values are extracted
    :return: grammar
    e.g., SELECT c1 FROM w WHERE c2 = (SELECT c3 FROM w WHERE c4 = v4) + 1
    """
    
    # db name
    db = "w"

    # extract column names
    fields = list(table.columns)

    # extract values per column
    value_per_column = {
        f"<{k}_values>": [str(x) for x in table[k].tolist() if pd.notnull(x)] or ["0"] for k in fields
    }

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            ['SELECT <SelList> FROM <Relation> WHERE <Comparison1>'],

        '<SelList>':
            ['<Attribute>'],

        '<Comparison1>':
            [f'{field2} = (SELECT {field3} FROM <Relation> WHERE {field4} = <{field4}_values>)' for field2, field3, field4 in zip(fields, fields, fields)],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    SQL_GRAMMAR.update(value_per_column)

    return SQL_GRAMMAR

def sql_grammar_in_clause2(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form:
    SELECT <field1> FROM <table> WHERE <field2> IN (<value1>, <value2>, <value3>).
    The output query is guaranteed to work on the provided table.
    :param table: table from which the fields and field values are extracted
    :return: grammar
    e.g., SELECT c1 FROM w WHERE c2 IN (v1, v2, v3)
    """

    # db name
    db = "w"

    # extract column names
    fields = list(table.columns)

    number_of_in = random.randint(1,3)
    # extract values per column
    value_per_column = {
        f"<{k}_values>": [str(x) for x in table[k].tolist() if pd.notnull(x)] or ["0"] for k in fields
    }

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            ['SELECT <SelList> FROM <Relation> WHERE <Condition>'],

        '<SelList>':
            ['<Attribute>'],

        '<Condition>':
            ['<Comparison>'],

        '<Comparison>':
            [f'{field} IN (<{field}_values>, <{field}_values>, <{field}_values>)' for field in fields],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    SQL_GRAMMAR.update(value_per_column)

    return SQL_GRAMMAR



def sql_grammar_in_clause(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form:
    SELECT <field1> FROM <table> WHERE <field2> IN (<value1>, <value2>, <value3>).
    The IN clause can have 1, 2, or 3 values.
    The output query is guaranteed to work on the provided table.
    :param table: table from which the fields and field values are extracted
    :return: grammar
    e.g., SELECT c1 FROM w WHERE c2 IN (v1, v2, v3)
    """

    # db name
    db = "w"

    # extract column names
    fields = list(table.columns)

    # extract values per column
    value_per_column = {
        f"<{k}_values>": [str(x) for x in table[k].tolist() if pd.notnull(x)] or ["0"] for k in fields
    }

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            ['SELECT <SelList> FROM <Relation> WHERE <Condition>'],

        '<SelList>':
            ['<Attribute>'],

        '<Condition>':
            ['<Comparison>'],

        '<Comparison>': [
            f'{field} IN (<{field}_values>)' for field in fields
        ] + [
            f'{field} IN (<{field}_values>, <{field}_values>)' for field in fields
        ] + [
            f'{field} IN (<{field}_values>, <{field}_values>, <{field}_values>)' for field in fields
        ],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    SQL_GRAMMAR.update(value_per_column)

    return SQL_GRAMMAR



def sql_grammar_WHERE_random_comparator(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form:
    SELECT <field> FROM w WHERE <field> = <field_value> or <field> != <field_value>.
    The output query is guaranteed to be working on the provided table.
    The comparison operator is chosen randomly between '=' and '!='.
    
    :param table: table from which the fields and field values are extracted
    :return: grammar
    """

    # db name
    db = "w"

    # extract column names
    fields = list(table.columns)

    # extract values per column
    value_per_column = {
        f"<{k}_values>": [str(x) for x in table[k].tolist() if x != ""] or ["0"] for k in fields
    }

    # randomly choose between '=' and '!=' for each field
    comparators = random.choice(['=', '!='])

    SQL_GRAMMAR = {
        '<Query>':
            ['SELECT <SelList> FROM <FromList> WHERE <Condition>'],

        '<SelList>':
            ['<Attribute>'],

        '<FromList>':
            ['<Relation>'],

        '<Condition>':
            ['<Comparison>'],

        '<Comparison>':
            [f'{field} {comparators} <{field}_values>' for field in fields],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    SQL_GRAMMAR.update(value_per_column)

    return SQL_GRAMMAR




def sql_grammar_AND_OR_random_comparator(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form: 
    SELECT <field> FROM w WHERE <field1> =/!= <value1> AND/OR <field2> =/!= <value2>.
    The output query is guaranteed to work on the provided table.
    The logical operator (AND/OR) and comparison operator (=/!=) are randomly chosen.
    
    :param table: table from which the fields and field values are extracted
    :return: grammar
    e.g., SELECT c1 FROM w WHERE c2 = v2 AND c3 != v3 or SELECT c1 FROM w WHERE c2 != v2 OR c3 = v3
    """

    # db name
    db = "w"
    
    # extract column names
    fields = list(table.columns)
    
    # extract values per column
    value_per_column = {
        f"<{k}_values>": [str(x) for x in table[k].tolist() if pd.notnull(x)] or ["0"] for k in fields
    }

    # randomly choose between 'AND' and 'OR'
    logical_operator = random.choice(['AND', 'OR'])

    # randomly choose between '=' and '!=' for each comparison
    comparator1 = random.choice(['=', '!='])
    comparator2 = random.choice(['=', '!='])

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            ['SELECT <SelList> FROM <FromList> WHERE <Condition>'],

        '<SelList>':
            ['<Attribute>'],

        '<FromList>':
            ['<Relation>'],

        '<Condition>':
            [f'<Comparison1> {logical_operator} <Comparison2>'],

        '<Comparison1>':
            [f'{field} {comparator1} <{field}_values>' for field in fields],

        '<Comparison2>':
            [f'{field} {comparator2} <{field}_values>' for field in fields],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    SQL_GRAMMAR.update(value_per_column)

    return SQL_GRAMMAR


def sql_grammar_WHEREOR2_random_comparator(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form:
    SELECT <field> FROM w WHERE <field1> =/!= <value1> AND/OR <field2> =/!= <value2> AND/OR <field3> =/!= <value3>.
    The output query is guaranteed to work on the provided table.
    The logical operators (AND/OR) and comparison operators (=/!=) are randomly chosen.
    
    :param table: table from which the field and field values are extracted
    :return: grammar
    e.g., SELECT c1 FROM w WHERE c2 = v2 OR c3 != v3 AND c4 = v4
    """

    # db name
    db = "w"

    # extract column names
    fields = list(table.columns)

    # extract values per column
    value_per_column = {
        f"<{k}_values>": [str(x) for x in table[k].tolist() if pd.notnull(x)] for k in fields
    }

    # randomly choose comparison operators for each condition
    comparator1 = random.choice(['=', '!='])
    comparator2 = random.choice(['=', '!='])
    comparator3 = random.choice(['=', '!='])

    # randomly choose logical operators
    logical_operator1 = random.choice(['AND', 'OR'])
    logical_operator2 = random.choice(['AND', 'OR'])

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            ['SELECT <SelList> FROM <FromList> WHERE <Condition>'],

        '<SelList>':
            ['<Attribute>'],

        '<FromList>':
            ['<Relation>'],

        '<Condition>':
            [f'<Comparison1> {logical_operator1} <Comparison2> {logical_operator2} <Comparison3>'],

        '<Comparison1>':
            [f'{field} {comparator1} <{field}_values>' for field in fields],

        '<Comparison2>':
            [f'{field} {comparator2} <{field}_values>' for field in fields],

        '<Comparison3>':
            [f'{field} {comparator3} <{field}_values>' for field in fields],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    SQL_GRAMMAR.update(value_per_column)

    return SQL_GRAMMAR




def sql_grammar_WHEREOR3_random_comparator(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form:
    SELECT <field> FROM w WHERE <field1> =/!= <value1> AND/OR <field2> =/!= <value2> AND/OR <field3> =/!= <value3> AND/OR <field4> =/!= <value4>.
    The output query is guaranteed to work on the provided table.
    The logical operators (AND/OR) and comparison operators (=/!=) are randomly chosen.
    
    :param table: table from which the field and field values are extracted
    :return: grammar
    e.g., SELECT c1 FROM w WHERE c2 = v2 OR c3 != v3 AND c4 = v4 OR c5 != v5
    """

    # db name
    db = "w"

    # extract column names
    fields = list(table.columns)

    # extract values per column
    value_per_column = {
        f"<{k}_values>": [str(x) for x in table[k].tolist() if pd.notnull(x)] for k in fields
    }

    # randomly choose comparison operators for each condition
    comparator1 = random.choice(['=', '!='])
    comparator2 = random.choice(['=', '!='])
    comparator3 = random.choice(['=', '!='])
    comparator4 = random.choice(['=', '!='])

    # randomly choose logical operators
    logical_operator1 = random.choice(['AND', 'OR'])
    logical_operator2 = random.choice(['AND', 'OR'])
    logical_operator3 = random.choice(['AND', 'OR'])

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            ['SELECT <SelList> FROM <FromList> WHERE <Condition>'],

        '<SelList>':
            ['<Attribute>'],

        '<FromList>':
            ['<Relation>'],

        '<Condition>':
            [f'<Comparison1> {logical_operator1} <Comparison2> {logical_operator2} <Comparison3> {logical_operator3} <Comparison4>'],

        '<Comparison1>':
            [f'{field} {comparator1} <{field}_values>' for field in fields],

        '<Comparison2>':
            [f'{field} {comparator2} <{field}_values>' for field in fields],

        '<Comparison3>':
            [f'{field} {comparator3} <{field}_values>' for field in fields],

        '<Comparison4>':
            [f'{field} {comparator4} <{field}_values>' for field in fields],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    SQL_GRAMMAR.update(value_per_column)

    return SQL_GRAMMAR



def sql_grammar_nested_select_random_comparator(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form:
    SELECT <field1> FROM <table> WHERE <field2> =/!= (SELECT <field3> FROM <table> WHERE <field4> = <value>) + 1.
    The comparison operator (= or !=) is randomly chosen.
    
    :param table: table from which the fields and field values are extracted
    :return: grammar
    """

    # db name
    db = "w"

    # extract column names
    fields = list(table.columns)

    # extract values per column
    value_per_column = {
        f"<{k}_values>": [str(x) for x in table[k].tolist() if pd.notnull(x)] or ["0"] for k in fields
    }

    # randomly choose the comparison operator between '=' and '!='
    comparator = random.choice(['=', '!='])

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            ['SELECT <SelList> FROM <Relation> WHERE <Comparison1>'],

        '<SelList>':
            ['<Attribute>'],

        '<Comparison1>':
            [f'{field2} {comparator} (SELECT {field3} FROM <Relation> WHERE {field4} = <{field4}_values>)'
             for field2, field3, field4 in zip(fields, fields, fields)],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    SQL_GRAMMAR.update(value_per_column)

    return SQL_GRAMMAR



def sql_grammar_in_clause_modified(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form:
    SELECT <field1> FROM <table> WHERE <field2> IN (<value1>, <value2>) or IN (<value1>, <value2>, <value3>) LIMIT k.
    The IN clause can have 2 or 3 values, and k is randomly chosen from 1, 2, or 3.
    
    :param table: table from which the fields and field values are extracted
    :return: grammar
    e.g., SELECT c1 FROM w WHERE c2 IN (v1, v2) LIMIT 2
    """

    # db name
    db = "w"

    # extract column names
    fields = list(table.columns)

    # extract values per column
    value_per_column = {
        f"<{k}_values>": [str(x) for x in table[k].tolist() if pd.notnull(x)] or ["0"] for k in fields
    }

    # Randomly choose the LIMIT value between 1, 2, or 3
    limit_value = random.choice([1, 2, 3])

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            [f'SELECT <SelList> FROM <Relation> WHERE <Condition> LIMIT {limit_value}'],

        '<SelList>':
            ['<Attribute>'],

        '<Condition>':
            ['<Comparison>'],

        # Only 2 or 3 values in the IN clause
        '<Comparison>': [
            f'{field} IN (<{field}_values>, <{field}_values>)' for field in fields
        ] + [
            f'{field} IN (<{field}_values>, <{field}_values>, <{field}_values>)' for field in fields
        ],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    SQL_GRAMMAR.update(value_per_column)

    return SQL_GRAMMAR



def sql_grammar_SELECT_modified(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form: SELECT <field> FROM w LIMIT k.
    The input table is ignored, and k is randomly chosen from 1, 2, or 3.
    
    :param table: ignored
    :return: grammar
    e.g., SELECT c1 FROM w LIMIT 2
    """
    db = "w"

    # extract column names
    fields = table.columns.tolist()

    # Randomly choose the LIMIT value between 1, 2, or 3
    limit_value = random.choice([1, 2, 3])

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            [f'SELECT <Attribute> FROM <Relation> LIMIT {limit_value}'],

        '<Attribute>': fields,

        '<Relation>': [db],
    }

    return SQL_GRAMMAR



def sql_grammar_AND_OR_random_comparator_with_limit(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form: 
    SELECT <field> FROM w WHERE <field1> =/!= <value1> AND/OR <field2> =/!= <value2> LIMIT k.
    The output query is guaranteed to work on the provided table.
    The logical operator (AND/OR) and comparison operator (=/!=) are randomly chosen, and k is randomly chosen from 1, 2, or 3.
    
    :param table: table from which the fields and field values are extracted
    :return: grammar
    e.g., SELECT c1 FROM w WHERE c2 = v2 AND c3 != v3 LIMIT 2 or SELECT c1 FROM w WHERE c2 != v2 OR c3 = v3 LIMIT 1
    """

    # db name
    db = "w"
    
    # extract column names
    fields = list(table.columns)
    
    # extract values per column
    value_per_column = {
        f"<{k}_values>": [str(x) for x in table[k].tolist() if pd.notnull(x)] or ["0"] for k in fields
    }

    # randomly choose between 'AND' and 'OR'
    logical_operator = random.choice(['AND', 'OR'])

    # randomly choose between '=' and '!=' for each comparison
    comparator1 = random.choice(['=', '!='])
    comparator2 = random.choice(['=', '!='])

    # randomly choose the LIMIT value between 1, 2, or 3
    limit_value = random.choice([1, 2, 3])

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            [f'SELECT <SelList> FROM <FromList> WHERE <Condition> LIMIT {limit_value}'],

        '<SelList>':
            ['<Attribute>'],

        '<FromList>':
            ['<Relation>'],

        '<Condition>':
            [f'<Comparison1> {logical_operator} <Comparison2>'],

        '<Comparison1>':
            [f'{field} {comparator1} <{field}_values>' for field in fields],

        '<Comparison2>':
            [f'{field} {comparator2} <{field}_values>' for field in fields],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    SQL_GRAMMAR.update(value_per_column)

    return SQL_GRAMMAR



def sql_grammar_WHEREOR2_random_comparator_with_limit(table: pd.DataFrame) -> Grammar:
    """
    Grammar generating SQL query of the form:
    SELECT <field> FROM w WHERE <field1> =/!= <value1> AND/OR <field2> =/!= <value2> AND/OR <field3> =/!= <value3> LIMIT k.
    The output query is guaranteed to work on the provided table.
    The logical operators (AND/OR) and comparison operators (=/!=) are randomly chosen, and k is randomly chosen from 1, 2, or 3.
    
    :param table: table from which the field and field values are extracted
    :return: grammar
    e.g., SELECT c1 FROM w WHERE c2 = v2 OR c3 != v3 AND c4 = v4 LIMIT 2
    """

    # db name
    db = "w"

    # extract column names
    fields = list(table.columns)

    # extract values per column
    value_per_column = {
        f"<{k}_values>": [str(x) for x in table[k].tolist() if pd.notnull(x)] for k in fields
    }

    # randomly choose comparison operators for each condition
    comparator1 = random.choice(['=', '!='])
    comparator2 = random.choice(['=', '!='])
    comparator3 = random.choice(['=', '!='])

    # randomly choose logical operators
    logical_operator1 = random.choice(['AND', 'OR'])
    logical_operator2 = random.choice(['AND', 'OR'])

    # randomly choose the LIMIT value between 1, 2, or 3
    limit_value = random.choice([1, 2, 3])

    SQL_GRAMMAR: Grammar = {
        '<Query>':
            [f'SELECT <SelList> FROM <FromList> WHERE <Condition> LIMIT {limit_value}'],

        '<SelList>':
            ['<Attribute>'],

        '<FromList>':
            ['<Relation>'],

        '<Condition>':
            [f'<Comparison1> {logical_operator1} <Comparison2> {logical_operator2} <Comparison3>'],

        '<Comparison1>':
            [f'{field} {comparator1} <{field}_values>' for field in fields],

        '<Comparison2>':
            [f'{field} {comparator2} <{field}_values>' for field in fields],

        '<Comparison3>':
            [f'{field} {comparator3} <{field}_values>' for field in fields],

        '<Relation>': [db],

        '<Attribute>': fields,
    }

    SQL_GRAMMAR.update(value_per_column)

    return SQL_GRAMMAR

#["IN", "LIMIT", "SELECT" ,"WHERE", "CONDI1", "CONDI2", "CONDI3", 'NESTEDSELECT']
#["IN_compo", "SELECT_compo", "CONDI1_compo", "CONDI2_compo"]

grammars : Dict = {
    "grammar_WHERE_composition" : sql_grammar_WHERE_composition,
    "WHERE" : sql_grammar_WHERE_random_comparator,
    "NOTWHERE" : sql_grammar_NOTWHERE,
    "OR1" : sql_grammar_WHEREOR1,
    "OR2" : sql_grammar_WHEREOR2,
    "SELECT" : sql_grammar_SELECT,
    "SUM" : sql_grammar_SUM,
    "GB" : sql_grammar_GB,
    "OB" : sql_grammar_OB,
    "L1" : sql_grammar_L1,
    "L2" : sql_grammar_L2,
    "L3" : sql_grammar_L3,
    "AND1" : sql_grammar_AND,
    "NESTEDSELECT" : sql_grammar_nested_select_random_comparator,
    "IN": sql_grammar_in_clause,
    "LIMIT":sql_grammar_LIMIT,
    "CONDI1":sql_grammar_AND_OR_random_comparator,
    "CONDI2":sql_grammar_WHEREOR2_random_comparator,
    "CONDI3":sql_grammar_WHEREOR3_random_comparator,
    "IN_compo":sql_grammar_in_clause_modified,
    "SELECT_compo":sql_grammar_SELECT_modified,
    "CONDI1_compo":sql_grammar_AND_OR_random_comparator_with_limit,
    "CONDI2_compo":sql_grammar_WHEREOR2_random_comparator_with_limit,
}

#["IN_compo", "SELECT_compo", "CONDI1_compo", "CONDI2_compo"]

