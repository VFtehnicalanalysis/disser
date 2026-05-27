def calculate_loan_payment_with_grace_period(loan_amount, annual_interest_rate, loan_term_years, grace_period_months=0, grace_interest_rate=0):
    monthly_interest_rate = annual_interest_rate / 100 / 12
    total_payments = loan_term_years * 12
    if grace_period_months > 0:
        monthly_payment = 0
        remaining_balance = loan_amount
        for month in range(grace_period_months):
            interest_payment = remaining_balance * (grace_interest_rate / 100 / 12)
            remaining_balance += interest_payment
        monthly_payment = remaining_balance * monthly_interest_rate / (1 - (1 + monthly_interest_rate) ** (-total_payments))
    else:
        monthly_payment = loan_amount * monthly_interest_rate / (1 - (1 + monthly_interest_rate) ** (-total_payments))
    absolute_grant_element = loan_amount - monthly_payment * total_payments
    relative_grant_element = absolute_grant_element / loan_amount
    print(f'Grace period (абсолютный грант-элемент):{absolute_grant_element:.2f}')
    print(f'Grace period (относительный грант-элемент):{relative_grant_element:.2f}')
    return (monthly_payment, absolute_grant_element, relative_grant_element)

def calculate_loan_payment_components(loan_amount, annual_interest_rate, loan_term_years, payment_structure='annuity'):
    monthly_interest_rate = annual_interest_rate / 100 / 12
    total_payments = loan_term_years * 12
    if payment_structure == 'annuity':
        monthly_payment = loan_amount * monthly_interest_rate / (1 - (1 + monthly_interest_rate) ** (-total_payments))
        total_payment_amount = monthly_payment * total_payments
        total_interest_amount = total_payment_amount - loan_amount
        interest_payments = [loan_amount * monthly_interest_rate for _ in range(total_payments)]
    elif payment_structure == 'differentiated':
        monthly_payment = loan_amount / total_payments
        total_interest_amount = 0
        principal_payments = []
        interest_payments = []
        remaining_balance = loan_amount
        for month in range(1, total_payments + 1):
            interest_payment = remaining_balance * monthly_interest_rate
            principal_payment = monthly_payment - interest_payment
            remaining_balance -= principal_payment
            interest_payments.append(interest_payment)
            principal_payments.append(principal_payment)
        total_payment_amount = sum(principal_payments) + sum(interest_payments)
    else:
        raise ValueError("Неверная структура срочной уплаты. Используйте 'annuity' или 'differentiated'.")
    return (monthly_payment, total_payment_amount, sum(interest_payments))
import pandas as pd
from jinja2 import Environment, FileSystemLoader

def create_payment_table(loan_amount, annual_interest_rate, loan_term_years, payment_structure='annuity'):
    months = list(range(1, loan_term_years * 12 + 1))
    total_payment = 0
    remaining_balance = loan_amount
    payments_data = {'Month': [], 'Monthly Payment ($)': [], 'Principal Payment ($)': [], 'Interest Payment ($)': [], 'Total Paid ($)': [], 'Remaining Balance ($)': []}
    for month in months:
        if payment_structure == 'annuity':
            monthly_payment, total_payment_amount, total_interest_amount = calculate_loan_payment_components(loan_amount, annual_interest_rate, loan_term_years, payment_structure=payment_structure)
            interest_payment = remaining_balance * (annual_interest_rate / 100 / 12)
            principal_payment = monthly_payment - interest_payment
            total_payment += monthly_payment
            remaining_balance -= principal_payment
        elif payment_structure == 'differentiated':
            monthly_payment, total_payment_amount, total_interest_amount = calculate_loan_payment_components(loan_amount, annual_interest_rate, loan_term_years, payment_structure=payment_structure)
            interest_payment = remaining_balance * (annual_interest_rate / 100 / 12)
            principal_payment = monthly_payment - interest_payment
            total_payment += monthly_payment
            remaining_balance -= principal_payment
        else:
            raise ValueError("Invalid payment structure. Use 'annuity' or 'differentiated'.")
        payments_data['Month'].append(month)
        payments_data['Monthly Payment ($)'].append(monthly_payment)
        payments_data['Principal Payment ($)'].append(principal_payment)
        payments_data['Interest Payment ($)'].append(interest_payment)
        payments_data['Total Paid ($)'].append(total_payment)
        payments_data['Remaining Balance ($)'].append(remaining_balance)
    payments_df = pd.DataFrame(payments_data)
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('payment_table_template.html')
    with open('payment_table.html', 'w') as file:
        file.write(template.render(table=payments_df.to_html(index=False)))
