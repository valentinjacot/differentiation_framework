import datetime


class Employee:
    # static variables:
    # They also can be changed for a single instance
    num_employee = 0
    raise_amount = 1.04

    def __init__(self, first_, last_):
        self.first = first_
        self.last = last_
        # self.email = first_.lower() + '.' + last_.lower() + '@company.com'
        Employee.num_employee += 1

    @property
    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    @property
    def email(self):
        return '{}.{}@company.com'.format(self.first, self.last)

    @fullname.setter
    def fullname(self, name):
        first, last = name.split(' ')
        self.first = first
        self.last = last

    @fullname.deleter
    def fullname(self):
        print('delete')
        self.first = None
        self.last = None



emp_1 = Employee('Corey', 'Schafer')

emp_1.first = 'jim'
emp_2 = Employee('Test', 'User')
print(emp_1.email)
print(emp_1.fullname)
emp_1.fullname = 'Valentin Jacot-Descombes'

print(emp_1.fullname)
del emp_1.fullname
print(emp_1.last)