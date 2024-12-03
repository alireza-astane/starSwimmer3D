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
    sphere { m*<-0.4456282262222225,-0.49103149663342827,-0.4559288890899455>, 1 }        
    sphere {  m*<0.9735392679779392,0.4989074172464891,9.393361207945201>, 1 }
    sphere {  m*<8.341326466300737,0.21381516645422693,-5.17731622112873>, 1 }
    sphere {  m*<-6.554636727388258,6.7368965400748655,-3.686509317947123>, 1}
    sphere { m*<-3.828614996386724,-7.85851612744561,-2.0225479420400783>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9735392679779392,0.4989074172464891,9.393361207945201>, <-0.4456282262222225,-0.49103149663342827,-0.4559288890899455>, 0.5 }
    cylinder { m*<8.341326466300737,0.21381516645422693,-5.17731622112873>, <-0.4456282262222225,-0.49103149663342827,-0.4559288890899455>, 0.5}
    cylinder { m*<-6.554636727388258,6.7368965400748655,-3.686509317947123>, <-0.4456282262222225,-0.49103149663342827,-0.4559288890899455>, 0.5 }
    cylinder {  m*<-3.828614996386724,-7.85851612744561,-2.0225479420400783>, <-0.4456282262222225,-0.49103149663342827,-0.4559288890899455>, 0.5}

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
    sphere { m*<-0.4456282262222225,-0.49103149663342827,-0.4559288890899455>, 1 }        
    sphere {  m*<0.9735392679779392,0.4989074172464891,9.393361207945201>, 1 }
    sphere {  m*<8.341326466300737,0.21381516645422693,-5.17731622112873>, 1 }
    sphere {  m*<-6.554636727388258,6.7368965400748655,-3.686509317947123>, 1}
    sphere { m*<-3.828614996386724,-7.85851612744561,-2.0225479420400783>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9735392679779392,0.4989074172464891,9.393361207945201>, <-0.4456282262222225,-0.49103149663342827,-0.4559288890899455>, 0.5 }
    cylinder { m*<8.341326466300737,0.21381516645422693,-5.17731622112873>, <-0.4456282262222225,-0.49103149663342827,-0.4559288890899455>, 0.5}
    cylinder { m*<-6.554636727388258,6.7368965400748655,-3.686509317947123>, <-0.4456282262222225,-0.49103149663342827,-0.4559288890899455>, 0.5 }
    cylinder {  m*<-3.828614996386724,-7.85851612744561,-2.0225479420400783>, <-0.4456282262222225,-0.49103149663342827,-0.4559288890899455>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    