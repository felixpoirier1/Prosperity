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
-H 'Authorization: Bearer eyJraWQiOiJ4M3NhZjFZTkNsRGwyVDljemdCR01ybnVVMlJlNDNjb1E1UGxYMWgwb2tBPSIsImFsZyI6IlJTMjU2In0.eyJzdWIiOiI4MmZjNzIzNC0wMzU0LTQ5NTAtYjhkOC0zOTg4OThkZTNiNDciLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaXNzIjoiaHR0cHM6XC9cL2NvZ25pdG8taWRwLmV1LXdlc3QtMS5hbWF6b25hd3MuY29tXC9ldS13ZXN0LTFfek9mVngwcWl3IiwiY29nbml0bzp1c2VybmFtZSI6IjgyZmM3MjM0LTAzNTQtNDk1MC1iOGQ4LTM5ODg5OGRlM2I0NyIsIm9yaWdpbl9qdGkiOiIxMzAxOTczMy0xMzYwLTQyYjktYTVhMC0wMWNhZGEwZDkyODciLCJhdWQiOiIzMmM1ZGM1dDFrbDUxZWRjcXYzOWkwcjJzMiIsImV2ZW50X2lkIjoiZjMyZmM4NWQtNzBhZC00Y2E1LWJkZWQtM2E4ZDY2MjY2OTk0IiwidG9rZW5fdXNlIjoiaWQiLCJhdXRoX3RpbWUiOjE3MTA4ODUxMzQsImV4cCI6MTcxMTI0NDY1NiwiaWF0IjoxNzExMjQxMDU2LCJqdGkiOiI1NTg5MWY3MC00YmY3LTQzNzMtOWIxOC0zMTdhYTMwMTViODAiLCJlbWFpbCI6ImZlbGl4cEB1Y2hpY2Fnby5lZHUifQ.af8KVPVCibbaUUtqpBoQOOiFeoQO41vTv0PacM38-e7gKBdA1qljs6oDFNl_TMnWlUupZpYbBmNDvebCZ5cT6VrWG4xqkJyBVk1nb-j0UxgT5ORsUn_yg1DiGrrJL9ZRqqAmJMYy_2m4dqPwZyu1YRkmepBsL9E8qVHVk8kDW5RATYuK999MwD8hI20iiW1m0iGd6GzDhjY1TX6IKgnbXOYcK1Kf8ZotQcW2NpfkLbzOJl3SvvdJ06-kcjtQyKPT6B71-gcz9LqTKCUdr0VXA2FAnocvJfZKjLYeJlgEjBOmWdXjpPagkE5u7j8QiQj-qxsXIWzg96I1ISZtFTJNVw' \
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
