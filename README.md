# drpeval
A place to put all our DRP testing scripts



## Quick start
### Install in conda or RPM release directory
Create a release directory. See for detail: [Psana Developer Documentation] (https://confluence.slac.stanford.edu/display/PSDMInternal/Psana+Developer+Documentation)

**on pslogin:**
```
cd <release-directory>
git clone https://github.com/lcls-psana/drpeval.git
```
### Do some work on psana nodes:
**put some code in**
- drpeval/src/ - .c, .cpp, .py files
- drpeval/include/ - .h file
- drpeval/app/ - python or cpp applications

**on psana node:**
```
scons # compiles everything automatically in sub-directories
# run apps
```
### Commit changes
```
git status
git add <updated-files>
git commit -m "comments on changes"
```
**on pslogin:**
```
git push origin master
```
