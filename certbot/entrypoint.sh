/wait-for-it.sh -t 30 nginx:80;
certbot certonly --webroot --webroot-path /var/www/certbot/ -d ${HOST_FQDN} --non-interactive --agree-tos -m info@${HOST_FQDN}
trap exit TERM;
while :;
do
  certbot renew; sleep 12h;
done;