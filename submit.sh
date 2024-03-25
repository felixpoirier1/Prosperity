echo "Enter the path of your file (default = trader.py): "
read filename

# Check if the filename is empty
if [ -z "$filename" ]; then
    # Set default filename
    filename="trader.py"
fi


curl -X POST 'https://bz97lt8b1e.execute-api.eu-west-1.amazonaws.com/prod/submission/algo' \
-H 'Accept: application/json, text/plain, */*' \
-H 'Accept-Encoding: gzip, deflate, br' \
-H 'Accept-Language: en,en-US;q=0.9,fr-FR;q=0.8,fr;q=0.7' \
-H 'Authorization: Bearer eyJraWQiOiJ4M3NhZjFZTkNsRGwyVDljemdCR01ybnVVMlJlNDNjb1E1UGxYMWgwb2tBPSIsImFsZyI6IlJTMjU2In0.eyJzdWIiOiI4MmZjNzIzNC0wMzU0LTQ5NTAtYjhkOC0zOTg4OThkZTNiNDciLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaXNzIjoiaHR0cHM6XC9cL2NvZ25pdG8taWRwLmV1LXdlc3QtMS5hbWF6b25hd3MuY29tXC9ldS13ZXN0LTFfek9mVngwcWl3IiwiY29nbml0bzp1c2VybmFtZSI6IjgyZmM3MjM0LTAzNTQtNDk1MC1iOGQ4LTM5ODg5OGRlM2I0NyIsIm9yaWdpbl9qdGkiOiIxMzAxOTczMy0xMzYwLTQyYjktYTVhMC0wMWNhZGEwZDkyODciLCJhdWQiOiIzMmM1ZGM1dDFrbDUxZWRjcXYzOWkwcjJzMiIsImV2ZW50X2lkIjoiZjMyZmM4NWQtNzBhZC00Y2E1LWJkZWQtM2E4ZDY2MjY2OTk0IiwidG9rZW5fdXNlIjoiaWQiLCJhdXRoX3RpbWUiOjE3MTA4ODUxMzQsImV4cCI6MTcxMTI0ODI1NywiaWF0IjoxNzExMjQ0NjU3LCJqdGkiOiJmMzVjZTc3MC0wNmNkLTRkYzgtYmRlMi1iNmMwYzRmMjExYzQiLCJlbWFpbCI6ImZlbGl4cEB1Y2hpY2Fnby5lZHUifQ.hrB6v4Yt-7ftszNIsP3B_PooVQpowIpOuR_UlSL1daR_kBloSqOzcNje2mntAMUdMdUHRxYZpNOt-5IbNZequaGvkHza2KBDAVPW2uMt4-2IE0AQDGduUpPVsKTngFxqxgJNxGqOXYb2gK5YcSQS1G7n0tDyzI1dd3sxylg0Z5UjlHnmUWSykuvY4jnnZSnHA9pr9GY64sem_Ns_VjDeC3CrvAquQSB-0QPpOpJLxHNPW-LmKT_Ci4s98Gt3xpKR-at9yVs1lDd_DGM2OpIFFgpiB1oUIffLZfsOliAuaSi45YILGdOrg42-7-s1N5IZBmbwIdXJ198vmOSPAooIqw' \
-H 'Content-Type: multipart/form-data' \
-H 'Origin: https://prosperity.imc.com' \
-H 'Referer: https://prosperity.imc.com/' \
-H 'Sec-Ch-Ua: "Chromium";v="122", "Not(A:Brand";v="24", "Brave";v="122"' \
-H 'Sec-Ch-Ua-Mobile: ?0' \
-H 'Sec-Ch-Ua-Platform: "macOS"' \
-H 'Sec-Fetch-Dest: empty' \
-H 'Sec-Fetch-Mode: cors' \
-H 'Sec-Fetch-Site: cross-site' \
-H 'Sec-Gpc: 1' \
-H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36' \
-F "file=@$PWD/$filename"