from prob_unit_test1 import titanic
import pandas as pd

from dtp import df_ready, get_attlab
from treeattfix import tdt_tit, p_d, t_d


# verify if the alteration of certain entries will maintain a constant result

def test_constant(tdt_tit, p_d):
    model = tdt_tit
    p13, p2 = p_d

    # Get masterpiece survival prob for first passenger=9%
    test_df = pd.DataFrame.from_dict([p13], orient='columns')
    X, y = get_attlab(df_ready(test_df))
    p13_prob = model.predict(X)[0]  

    # Alter name from Owen to Mary without altering sex or status= 0.09
    p13_name = p13.copy()
    p13_name['Name'] = ' Mr. Ade'
    test_df = pd.DataFrame.from_dict([p13_name], orient='columns')
    X, y = get_attlab(df_ready(test_df))
    p13_name_prob = model.predict(X)[0]  

    # alter ticket no 
    p13_ticket = p13.copy()
    p13_ticket['ticket'] = 'PC 17599'
    test_df = pd.DataFrame.from_dict([p13_ticket], orient='columns')
    X, y = get_attlab(df_ready(test_df))
    p13_ticket_prob = model.predict(X)[0] 

    # alter embarked port =0.09
    p13_port = p13.copy()
    p13_port['Embarked'] = 'C'
    test_df = pd.DataFrame.from_dict([p13_port], orient='columns')
    X, y = get_attlab(df_ready(test_df))
    p13_port_prob = model.predict(X)[0] 

    # Get masterpiece survival prob of the second passenger =1.0
    test_df = pd.DataFrame.from_dict([p2], orient='columns')
    X, y = get_attlab(df_ready(test_df))
    p2_prob = model.predict(X)[0] 

    # Alter the name without altering sex or status=1.0
    p2_name = p2.copy()
    p2_name['Name'] = ' Mrs. Berns'
    test_df = pd.DataFrame.from_dict([p2_name], orient='columns')
    X, y = get_attlab(df_ready(test_df))
    p2_name_prob = model.predict(X)[0]  

    # Alter ticket no = 1.0
    p2_ticket = p2.copy()
    p2_ticket['ticket'] = 'A/5 21171'
    test_df = pd.DataFrame.from_dict([p2_ticket], orient='columns')
    X, y = get_attlab(df_ready(test_df))
    p2_ticket_prob = model.predict(X)[0] 

    # Alter embarked port= 1.0
    p2_port = p2.copy()
    p2_port['Embarked'] = 'Q'
    test_df = pd.DataFrame.from_dict([p2_port], orient='columns')
    X, y = get_attlab(df_ready(test_df))
    p2_port_prob = model.predict(X)[0]  

    assert p13_prob == p13_name_prob == p13_ticket_prob == p13_port_prob
    assert p2_prob == p2_name_prob == p2_ticket_prob == p2_port_prob


# Verify if altering entry (e.g.sex, pclass) will influence survival probab in anticipated route
def test_dt_vary(tdt_tit, p_d):
    model = tdt_tit
    p13, p2 = p_d

    # input masterpiece survival prob of the first passenger=0.09
    test_df = pd.DataFrame.from_dict([p13], orient='columns')
    X, y = get_attlab(df_ready(test_df))
    p13_prob = model.predict(X)[0]  

    # Alter sex from masculine to feminine=0.65
    p13_female = p13.copy()
    p13_female['Name'] = ' Mrs. Owen'
    p13_female['Sex'] = 'female'
    test_df = pd.DataFrame.from_dict([p13_female], orient='columns')
    X, y = get_attlab(df_ready(test_df))
    p13_female_prob = model.predict(X)[0]

    # Alter pclass from 3 to 1=36%
    p13_class = p13.copy()
    p13_class['Pclass'] = 1
    test_df = pd.DataFrame.from_dict([p13_class], orient='columns')
    X, y = get_attlab(df_ready(test_df))
    p13_class_prob = model.predict(X)[0] 

    assert p13_prob < p13_female_prob, 'Altering sex from masculine to feminine ougth to augment likelihood for survival.'
    assert p13_prob < p13_class_prob, 'Altering class from 3 to 1 ought to augment  likelihood for survival.'

    # input masterpiece likelihood for survival of the second passenger=100%
    test_df = pd.DataFrame.from_dict([p2], orient='columns')
    X, y = get_attlab(df_ready(test_df))
    p2_prob = model.predict(X)[0] 

    # Alter sex from female to male=56%
    p2_male = p2.copy()
    p2_male['Name'] = ' Mr. John'
    p2_male['Sex'] = 'male'
    test_df = pd.DataFrame.from_dict([p2_male], orient='columns')
    X, y = get_attlab(df_ready(test_df))
    p2_male_prob = model.predict(X)[0] 

    # Alter pclass from 1 to 3=0%
    p2_class = p2.copy()
    p2_class['Pclass'] = 3
    test_df = pd.DataFrame.from_dict([p2_class], orient='columns')
    X, y = get_attlab(df_ready(test_df))
    p2_class_prob = model.predict(X)[0]  

    # reduce fare from 71.2833 to 5=85%
    p2_fare = p2.copy()
    p2_fare['Fare'] = 5
    test_df = pd.DataFrame.from_dict([p2_fare], orient='columns')
    X, y = get_attlab(df_ready(test_df))
    p2_fare_prob = model.predict(X)[0]  

    assert p2_prob > p2_male_prob, 'Altering sex from feminine to masculine ought to reduce the likelihood for survival.'
    assert p2_prob > p2_class_prob, 'Altering pclass from 1 to 3 ought to reduce the likelihood for survival.'
    assert p2_prob > p2_fare_prob, 'Altering fare from 72 to 5 ought to reduce the likelihood for survival.'