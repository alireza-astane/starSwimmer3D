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
    sphere { m*<-1.586534913389424,-0.19377430900198764,-1.0232505860149685>, 1 }        
    sphere {  m*<-0.12041302592398218,0.23973904949792557,8.859276171875589>, 1 }
    sphere {  m*<7.234938412075977,0.15081877350356743,-5.720217118169774>, 1 }
    sphere {  m*<-3.302703807074683,2.1766671202395442,-1.903580710063761>, 1}
    sphere { m*<-2.9880889636772947,-2.754206864050403,-1.7147183668833619>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.12041302592398218,0.23973904949792557,8.859276171875589>, <-1.586534913389424,-0.19377430900198764,-1.0232505860149685>, 0.5 }
    cylinder { m*<7.234938412075977,0.15081877350356743,-5.720217118169774>, <-1.586534913389424,-0.19377430900198764,-1.0232505860149685>, 0.5}
    cylinder { m*<-3.302703807074683,2.1766671202395442,-1.903580710063761>, <-1.586534913389424,-0.19377430900198764,-1.0232505860149685>, 0.5 }
    cylinder {  m*<-2.9880889636772947,-2.754206864050403,-1.7147183668833619>, <-1.586534913389424,-0.19377430900198764,-1.0232505860149685>, 0.5}

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
    sphere { m*<-1.586534913389424,-0.19377430900198764,-1.0232505860149685>, 1 }        
    sphere {  m*<-0.12041302592398218,0.23973904949792557,8.859276171875589>, 1 }
    sphere {  m*<7.234938412075977,0.15081877350356743,-5.720217118169774>, 1 }
    sphere {  m*<-3.302703807074683,2.1766671202395442,-1.903580710063761>, 1}
    sphere { m*<-2.9880889636772947,-2.754206864050403,-1.7147183668833619>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.12041302592398218,0.23973904949792557,8.859276171875589>, <-1.586534913389424,-0.19377430900198764,-1.0232505860149685>, 0.5 }
    cylinder { m*<7.234938412075977,0.15081877350356743,-5.720217118169774>, <-1.586534913389424,-0.19377430900198764,-1.0232505860149685>, 0.5}
    cylinder { m*<-3.302703807074683,2.1766671202395442,-1.903580710063761>, <-1.586534913389424,-0.19377430900198764,-1.0232505860149685>, 0.5 }
    cylinder {  m*<-2.9880889636772947,-2.754206864050403,-1.7147183668833619>, <-1.586534913389424,-0.19377430900198764,-1.0232505860149685>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    