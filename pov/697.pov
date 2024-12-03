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
    sphere { m*<3.86008113133047e-18,-5.443501600796865e-18,0.8636082116357643>, 1 }        
    sphere {  m*<5.308348733494946e-19,-5.246079553793385e-18,5.977608211635788>, 1 }
    sphere {  m*<9.428090415820634,4.1405640050398024e-20,-2.4697251216975684>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.4697251216975684>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.4697251216975684>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<5.308348733494946e-19,-5.246079553793385e-18,5.977608211635788>, <3.86008113133047e-18,-5.443501600796865e-18,0.8636082116357643>, 0.5 }
    cylinder { m*<9.428090415820634,4.1405640050398024e-20,-2.4697251216975684>, <3.86008113133047e-18,-5.443501600796865e-18,0.8636082116357643>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.4697251216975684>, <3.86008113133047e-18,-5.443501600796865e-18,0.8636082116357643>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.4697251216975684>, <3.86008113133047e-18,-5.443501600796865e-18,0.8636082116357643>, 0.5}

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
    sphere { m*<3.86008113133047e-18,-5.443501600796865e-18,0.8636082116357643>, 1 }        
    sphere {  m*<5.308348733494946e-19,-5.246079553793385e-18,5.977608211635788>, 1 }
    sphere {  m*<9.428090415820634,4.1405640050398024e-20,-2.4697251216975684>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.4697251216975684>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.4697251216975684>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<5.308348733494946e-19,-5.246079553793385e-18,5.977608211635788>, <3.86008113133047e-18,-5.443501600796865e-18,0.8636082116357643>, 0.5 }
    cylinder { m*<9.428090415820634,4.1405640050398024e-20,-2.4697251216975684>, <3.86008113133047e-18,-5.443501600796865e-18,0.8636082116357643>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.4697251216975684>, <3.86008113133047e-18,-5.443501600796865e-18,0.8636082116357643>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.4697251216975684>, <3.86008113133047e-18,-5.443501600796865e-18,0.8636082116357643>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    