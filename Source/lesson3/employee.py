
# Create Class as Employee
class Employee:
    # create class counter variable
    counter = 0
    # create a class new list to save the employee pays
    paylist = []

    # Constructor method
    def __init__(self, first, family, pay, department):
        self.first = first
        self.family = family
        self.pay = pay
        self.department = department

        ##Add the Salary to paylist
        Employee.paylist.append(pay)

        #add to class counter var. +1 whenever new object created
        Employee.counter +=1


    def fullname(self):
        return('{} {}'.format(self.first, self.family))

    def salaryavg():
        # total salary variable
        total = 0
        for x in Employee.paylist:
            total +=x
        # found salary avg
        ave = total / len(Employee.paylist)
        return ave

#Create Full Time Class and inheranit from Employee Class
class FulltimeEmployee(Employee):
    def em_rasise(self):
        raise_per = input('Enter Annual Raise %: ')
        raise_per = float(raise_per)
        self.pay = round((self.pay * raise_per),3)
        return self.pay


# Enter Employee information
employee1 = Employee('Rawa', 'Albattawi',55000,'it')
employee2 = Employee('Ali', 'Badri', 60000, 'HR')
employee3 = Employee('Dan', 'Badri',70000,'Network' )
employee4 = FulltimeEmployee('MD', 'Badri',70000,'HR')

#Print employee information
print('Employee Name: ' + employee1.fullname() +', Dep: '+ employee1.department +', Annual Salary: ' + str(employee1.pay))
print('Employee Name: ' + employee2.fullname() +', Dep: '+ employee2.department +', Annual Salary: ' + str(employee2.pay))
print('Employee Name: ' + employee3.fullname() +', Dep: '+ employee3.department +', Annual Salary: ' + str(employee3.pay))
print()
#print Full Time Employee
print('Full Time Employee info:\n-----------------------------')
print('Employee full name: ' + employee4.fullname() +
      '\nDep: ' + employee4.department +
      '\nAnnual Salary: ' + str(employee4.pay))
print('Salary Raise: ' + str(employee4.em_rasise()))
#print space
print()
#print total number of empoyee and the Avg Salary
print('Total Emplyoee number# ' + str(Employee.counter))
print('Salary Avg: ' + str(round(Employee.salaryavg(),2)))
