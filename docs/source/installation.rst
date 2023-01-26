Installing SURFACE
==================

Here we'll include information on how to install SURFACE.

S.U.R.F.A.C.E. CDMS installation involves the following six(6) steps
Training Doc

PLEASE NOTE - in an attempt to make the installation process easier, all linux commands are shown in ONE(1) line, to make copying/pasting easier. Therefore for long linux commands the font size will be smaller so that it can fit in one(1) line.(we had issues copying and pasting with multiline commands)


OS and Virtual Machine Versions

Operating System: 
Ubuntu 20.04.5 (Focal Fossa)
https://releases.ubuntu.com/20.04.5


Virtual Machine:
VirtualBox  7.0.2 (Windows hosts)
https://www.virtualbox.org/wiki/Downloads
Initial Setup/Updates - minimal machine specifications(for testing not production)
System Information:(if using a VM you can use these specs)
Main Memory:
4 GB (min)
16 GB (servers)
Disk storage:
25 GB (min)
100 GB (servers)
CPU cores:
4 (min)
8 (servers)


Update, Upgrade and Packages
$ sudo apt update
$ sudo apt upgrade
ASIDE - Virtual machine users may want to install openssh. Continue setup via ssh by following commands below
sudo apt install openssh-server
sudo systemctl status ssh
ctrl + C 
sudo ufw allow ssh




BACK TO SURFACE INSTALLATION
STEP 1 - Install important pre-requisities
$ sudo apt install build-essential gcc make perl dkms curl
$ sudo apt install git
Prerequisites (installation steps linked below and outlined in steps to follow)
Docker:
https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04
Docker-Compose:
https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-compose-on-ubuntu-20-04
PostgreSQL:(NOT NEEDED)
https://www.digitalocean.com/community/tutorials/how-to-install-postgresql-on-ubuntu-20-04-quickstart

Step 2 - Docker Installation
Update and Prerequisite Packages:
$ sudo apt update
$ sudo apt install apt-transport-https ca-certificates curl software-properties-common


Add the GPG key for the official Docker repository
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -


Add the Docker repository to APT sources:
$ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"


Make sure you are about to install from the Docker repo instead of the default Ubuntu repo:
$ apt-cache policy docker-ce


Install Docker
$ sudo apt install docker-ce
$ sudo systemctl status docker
ctrl+c




Setting docker to run without sudo:
$ sudo usermod -aG docker ${USER}
$ su - ${USER}
$ groups

PLEASE NOTE - for some reason during the installation process(my experience) we had to run these commands repeatedly to make sure that “docker” was a user within “groups”. When you run the “groups” command if “docker” is not listed some of the following installation steps failed so we had to run these commands again to ensure that “docker” was displayed under “groups”. I do not know why this was happening.


(VM) Restart your Virtual Machine(If you are running a VM)







Step 2 - Docker-Compose Installation
Download the 1.29.2 release and save the executable file at ‘/usr/local/bin/docker-compose’:
$ sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
ONE(1)line version below:
$ sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose


Setting Permissions for docker-compose
$ sudo chmod +x /usr/local/bin/docker-compose


Verifying Installation:
$ docker-compose --version


Step 3 - Download repository and production file 
Download the source code from GitHub repository
$ git clone https://github.com/opencdms/surface



Configure environment variables in ‘surface/api/production.env’ file.
(NOTE - for security purposes the production.env file needed to build the environment is not found in the repository. The repository contains a sample file that will need to be replaced. The actual file is provided to you below:
FTP Server details(production.env file located here)
Host:		db.hydromet.gov.bz:5394
User:		wxstation
Password: 	wxst@t!0n
Command is “get production.env”





STEP 4 - Build Docker images
Build Docker Images - 


$ docker-compose build
(NOTE -  The build command should be run from the “/surface” directory (you may currently be in the surface/api directory after setting up the production.env file, therefore you need to “cd ..” to go back to the “/surface” directory)


Start Docker - must start only with these 4 containers


$ docker-compose up postgres cache redis api


NOTE  - You will need to open a new terminal at this point to run the commands to follow. If you are using a terminal only installation you will need to “Ctrl + \” to exit docker without killing the containers 
Starting SURFACE database (had an error message here once - - problem noted in training doc)
Without Data:
$ docker-compose exec api bash load_initial_data.sh




With backup data dump file:
$ docker-compose exec -T postgres psql -U dba -d surface_db < backup_data.sql


STEP 5 - Initial setup to Postgres database
Collect Static Files and Create User:
$ docker-compose exec api python manage.py collectstatic --noinput

$ docker-compose exec api python manage.py createsuperuser


STEP 6 - Starting SURFACE
Stop Docker:
$ docker-compose stop (CTRL+C)


Start Docker
$ docker-compose up
(background start) $ docker-compose up -d


Open in Browser:
0.0.0.0:8080

Installation Questions


