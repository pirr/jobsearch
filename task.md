Job searching

Idea:
Searching relevant job for candidate.

How:
System read information about candidate - urls, pdfs, text files, doc files, excel and etc.
System found relevant source of vacancies for according the information.
Vacansies raiting for candidate - according candidate info and vacancy description.
System show list vacansies in order (from much fit to less fit), treschold is used to dismissed useless vacansies. System show vacancy link, short description and why it fit and unfit for candidate.

MVP:
Read candidate info from pdfs, text, doc
Rating: candidate stack and relevan experience against vacansy description (need to make first iteration simple but robust)
Use LLM - local or cloud

Arch:
main - DDD 
Documents reader (urls, pdfs, ...) - dependency injection, factory pattern
Searching - use background taks, different searches can adopt for different sources. Search and data storing different operations and should be separate and reliable. A lot different aproaches will be ther, need to think how to do this in best.
Rating - use LLM, different aproaches can be ther.

All module should be clear and resposible for one business domain. Don't mix different business logic!

Frontend:
Can plug different type terminal, web, phone (for MVP use terminal or web - choose one which be simple)

Backend - python.
