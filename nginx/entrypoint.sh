export DOLLAR="$"
cat /etc/nginx-conf-template/nginx.conf.http.template >| /etc/nginx/conf.d/nginx.conf
/wait-for-it.sh -t 30 api:8000;
echo $HOST_FQDN
if [ -z ${HOST_FQDN+x} ];
then
  nginx;
else
  nginx;
  certbot certonly --webroot --webroot-path /var/www/certbot/ -d ${HOST_FQDN} --non-interactive --agree-tos -m info@${HOST_FQDN}
  envsubst < /etc/nginx-conf-template/nginx.conf.https.template >| /etc/nginx/conf.d/nginx.conf
  nginx -s stop;
  nginx;
fi
while :;
do
  sleep 6h;
  nginx -s reload;
done