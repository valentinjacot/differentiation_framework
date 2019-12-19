import datetime


class Employee:
    # static variables:
    # They also can be changed for a single instance
    num_employee = 0
    raise_amount = 1.04

    def __init__(self, first_, last_, pay_):
        self.first = first_
        self.last = last_
        self.pay = pay_
        self.email = first_.lower() + '.' + last_.lower() + '@company.com'
        Employee.num_employee += 1

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    def apply_raise(self):
        # self.pay = int(self.pay * Employee.raise_amount) #bad
        self.pay = int(self.pay * self.raise_amount)  # better

    # Class method: do not take instance as argument, rather take the class it self
    # Can be used as an alternative to constructors
    @classmethod
    def set_raise_amt(cls, amt_):
        cls.raise_amount = amt_

    @classmethod
    def from_string(cls, emp_string):
        first_, last_, pay_ = emp_string.split('-')
        return cls(first_, last_, pay_)

    @staticmethod
    def is_workday(day):
        if day.weekday() == 5 or day.weekday() == 6:
            return False
        return True

    def __repr__(self):
        return "Employee( '{}', '{}', '{}')".format(self.first, self.last, self.pay)

    def __str__(self):
        return "Employee( '{}' - '{}')".format(self.fullname(), self.email)

    def __add__(self, other):
        return self.pay + other.pay

    def __len__(self):
        return self.fullname().__len__()
        




class Developer(Employee):
    raise_amount = 1.1

    def __init__(self, first_, last_, pay_, prog_lang_):
        super().__init__(first_, last_, pay_)
        self.prog_lang = prog_lang_


class Manager(Employee):
    raise_amount = 1.3

    def __init__(self, first_, last_, pay_, employees_=None):
        super().__init__(first_, last_, pay_)
        if employees_ is None:
            self.employees = []
        else:
            self.employees = employees_

    def add_emp(self, emp_):
        if emp_ not in self.employees:
            self.employees.append(emp_)

    def remove_emp(self, emp_):
        if emp_ in self.employees:
            self.employees.remove(emp_)

    def print_employees(self):
        print(self.fullname() + ' manages : ')
        for emp in self.employees:
            print('-->' + emp.fullname())


emp_1 = Employee('Corey', 'S.', 50000)
emp_2 = Employee('Test', 'User', 60000)

emp_3 = Developer('Val', 'Jacot', 100000, 'python')
# print(emp_3.prog_lang)
# print(emp_3.email)
emp_4 = Manager('Valentin', 'Jacot-Descombes', 100000, [emp_1, emp_2])
# print(emp_4.email)
# emp_4.add_emp(emp_3)
#
# print(emp_4.print_employees())
#
# print(issubclass(Developer,Manager))
# dunder = '__'
print(repr(emp_1))
print(str(emp_1))
print(emp_4.__str__())

print(emp_1+ emp_4)

print(emp_4.last.__len__())
print(len(emp_4.last))
print(len(emp_4))
# print(Employee.num_employee)
#
# emp_1.raise_amount = 1.05
# Employee.set_raise_amt(1.01)
# # print(emp_1.pay)pay
# # emp_1.apply_raise()
# print(emp_1.raise_amount)
# print(emp_2.raise_amount)
# print(Employee.raise_amount)
#
# # print(emp_2.email)k
# # print(Employee.fullname(emp_1))
#
# emp_str_1 = 'John-Doe-70000'
#
# emp_3 = Employee.from_string(emp_str_1)
# print(emp_3.email)
# myDate = datetime.date(2019,11,1)
# print(Employee.is_workday(myDate))
