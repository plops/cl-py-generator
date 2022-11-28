
# i develop inside arch linux container inside a termux environment on
# an android phone.  i need the arch linux because sbcl doesn't run in
# termux.  unfortunately nbdev doesn't work in the arch linux
# container

# in order to stream line the process i want to be able to call nbdev
# from inside the arch linux container using ssh. this script creates
# the ssh key

ssh-keygen -t ed25519 \
	   -a 420 \
	   -f ~/.ssh/arch_to_termux.ed25519 \
	   -C "internal connection on C11 Android device"


# modify ~/.ssh/config on arch linux lke this

#Host c11
#  HostName localhost
#  Port 8022
#  User u0_a221
#  IdentitiesOnly yes
#  IdentityFile ~/.ssh/arch_to_termux.ed25519

# on termux enable key like so:
# cat ~/arch/home/martin/.ssh/arch_to_termux.ed25519.pub >> ~/.ssh/authorized_keys
# everytime you start termux you have to manually start the sshd service. AFAIK you can't have the service start automatically
# sshd
