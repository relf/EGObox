{%- set file_content = load_data(url=url, format="plain") -%}

<div class="code-wrapper">


```python,copy
{{ file_content | safe }}
```


</div>

