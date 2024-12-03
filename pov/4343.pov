#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.1845373781126987,-0.09388151660038463,-0.6458343692678884>, 1 }        
    sphere {  m*<0.24942777070165706,0.1381398509993482,4.739731031472089>, 1 }
    sphere {  m*<2.550171015893558,0.008152458785989553,-1.8750438947190704>, 1 }
    sphere {  m*<-1.8061527380055886,2.2345924278182148,-1.6197801346838572>, 1}
    sphere { m*<-1.5383655169677568,-2.6530995145856826,-1.4302338495212843>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.24942777070165706,0.1381398509993482,4.739731031472089>, <-0.1845373781126987,-0.09388151660038463,-0.6458343692678884>, 0.5 }
    cylinder { m*<2.550171015893558,0.008152458785989553,-1.8750438947190704>, <-0.1845373781126987,-0.09388151660038463,-0.6458343692678884>, 0.5}
    cylinder { m*<-1.8061527380055886,2.2345924278182148,-1.6197801346838572>, <-0.1845373781126987,-0.09388151660038463,-0.6458343692678884>, 0.5 }
    cylinder {  m*<-1.5383655169677568,-2.6530995145856826,-1.4302338495212843>, <-0.1845373781126987,-0.09388151660038463,-0.6458343692678884>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.1845373781126987,-0.09388151660038463,-0.6458343692678884>, 1 }        
    sphere {  m*<0.24942777070165706,0.1381398509993482,4.739731031472089>, 1 }
    sphere {  m*<2.550171015893558,0.008152458785989553,-1.8750438947190704>, 1 }
    sphere {  m*<-1.8061527380055886,2.2345924278182148,-1.6197801346838572>, 1}
    sphere { m*<-1.5383655169677568,-2.6530995145856826,-1.4302338495212843>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.24942777070165706,0.1381398509993482,4.739731031472089>, <-0.1845373781126987,-0.09388151660038463,-0.6458343692678884>, 0.5 }
    cylinder { m*<2.550171015893558,0.008152458785989553,-1.8750438947190704>, <-0.1845373781126987,-0.09388151660038463,-0.6458343692678884>, 0.5}
    cylinder { m*<-1.8061527380055886,2.2345924278182148,-1.6197801346838572>, <-0.1845373781126987,-0.09388151660038463,-0.6458343692678884>, 0.5 }
    cylinder {  m*<-1.5383655169677568,-2.6530995145856826,-1.4302338495212843>, <-0.1845373781126987,-0.09388151660038463,-0.6458343692678884>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    