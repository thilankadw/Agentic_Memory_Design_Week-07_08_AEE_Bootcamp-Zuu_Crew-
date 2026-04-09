-- Retrive all table names

SELECT 'table' AS obj_type, table_name AS obj_name
FROM information_schema.tables
WHERE table_schema = 'public' AND table_type = 'BASE TABLE';

------------------------------------------------
----------------- CRM TABLES -------------------
------------------------------------------------
SELECT * FROM patients LIMIT 100;
SELECT * FROM specialties LIMIT 100;
SELECT * FROM locations LIMIT 100;
SELECT * FROM doctors LIMIT 100;
SELECT * FROM bookings LIMIT 100;


SELECT COUNT(*) FROM patients;
SELECT COUNT(*) FROM specialties;
SELECT COUNT(*) FROM locations;
SELECT COUNT(*) FROM doctors;
SELECT COUNT(*) FROM bookings;

------------------------------------------------
----------------- MEMORY TABLES -------------------
------------------------------------------------
SELECT * FROM st_turns;
SELECT * FROM mem_facts;
SELECT * FROM mem_episodes;
SELECT * FROM mem_procedures;