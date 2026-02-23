
from dotenv import load_dotenv
from clients import get_redis

load_dotenv()

r = get_redis()

success = r.set('foo', 'bar')
result = r.get('foo')
print(result)
r.close()
