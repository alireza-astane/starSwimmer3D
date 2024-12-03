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
    sphere { m*<-1.3655252897552503,-0.17809954714088838,-1.153257577661593>, 1 }        
    sphere {  m*<0.007159303131049677,0.2796553419366851,8.741460501782342>, 1 }
    sphere {  m*<6.502193723267158,0.09390860163363432,-5.246685314241183>, 1 }
    sphere {  m*<-3.035326640211829,2.1511062220402937,-2.040135558060362>, 1}
    sphere { m*<-2.7675394191739975,-2.7365857203636037,-1.8505892728977917>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.007159303131049677,0.2796553419366851,8.741460501782342>, <-1.3655252897552503,-0.17809954714088838,-1.153257577661593>, 0.5 }
    cylinder { m*<6.502193723267158,0.09390860163363432,-5.246685314241183>, <-1.3655252897552503,-0.17809954714088838,-1.153257577661593>, 0.5}
    cylinder { m*<-3.035326640211829,2.1511062220402937,-2.040135558060362>, <-1.3655252897552503,-0.17809954714088838,-1.153257577661593>, 0.5 }
    cylinder {  m*<-2.7675394191739975,-2.7365857203636037,-1.8505892728977917>, <-1.3655252897552503,-0.17809954714088838,-1.153257577661593>, 0.5}

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
    sphere { m*<-1.3655252897552503,-0.17809954714088838,-1.153257577661593>, 1 }        
    sphere {  m*<0.007159303131049677,0.2796553419366851,8.741460501782342>, 1 }
    sphere {  m*<6.502193723267158,0.09390860163363432,-5.246685314241183>, 1 }
    sphere {  m*<-3.035326640211829,2.1511062220402937,-2.040135558060362>, 1}
    sphere { m*<-2.7675394191739975,-2.7365857203636037,-1.8505892728977917>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.007159303131049677,0.2796553419366851,8.741460501782342>, <-1.3655252897552503,-0.17809954714088838,-1.153257577661593>, 0.5 }
    cylinder { m*<6.502193723267158,0.09390860163363432,-5.246685314241183>, <-1.3655252897552503,-0.17809954714088838,-1.153257577661593>, 0.5}
    cylinder { m*<-3.035326640211829,2.1511062220402937,-2.040135558060362>, <-1.3655252897552503,-0.17809954714088838,-1.153257577661593>, 0.5 }
    cylinder {  m*<-2.7675394191739975,-2.7365857203636037,-1.8505892728977917>, <-1.3655252897552503,-0.17809954714088838,-1.153257577661593>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    