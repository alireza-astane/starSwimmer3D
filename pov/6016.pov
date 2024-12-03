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
    sphere { m*<-1.5813265245611812,-0.2031240302112651,-1.0205657050683812>, 1 }        
    sphere {  m*<-0.11549902846450211,0.23543995757384723,8.861781661842524>, 1 }
    sphere {  m*<7.2398524095354615,0.14651968157948925,-5.71771162820284>, 1 }
    sphere {  m*<-3.3347679438779165,2.2144617535064204,-1.9199887244990195>, 1}
    sphere { m*<-2.9808786560202947,-2.7649261992373044,-1.7110151047711164>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.11549902846450211,0.23543995757384723,8.861781661842524>, <-1.5813265245611812,-0.2031240302112651,-1.0205657050683812>, 0.5 }
    cylinder { m*<7.2398524095354615,0.14651968157948925,-5.71771162820284>, <-1.5813265245611812,-0.2031240302112651,-1.0205657050683812>, 0.5}
    cylinder { m*<-3.3347679438779165,2.2144617535064204,-1.9199887244990195>, <-1.5813265245611812,-0.2031240302112651,-1.0205657050683812>, 0.5 }
    cylinder {  m*<-2.9808786560202947,-2.7649261992373044,-1.7110151047711164>, <-1.5813265245611812,-0.2031240302112651,-1.0205657050683812>, 0.5}

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
    sphere { m*<-1.5813265245611812,-0.2031240302112651,-1.0205657050683812>, 1 }        
    sphere {  m*<-0.11549902846450211,0.23543995757384723,8.861781661842524>, 1 }
    sphere {  m*<7.2398524095354615,0.14651968157948925,-5.71771162820284>, 1 }
    sphere {  m*<-3.3347679438779165,2.2144617535064204,-1.9199887244990195>, 1}
    sphere { m*<-2.9808786560202947,-2.7649261992373044,-1.7110151047711164>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.11549902846450211,0.23543995757384723,8.861781661842524>, <-1.5813265245611812,-0.2031240302112651,-1.0205657050683812>, 0.5 }
    cylinder { m*<7.2398524095354615,0.14651968157948925,-5.71771162820284>, <-1.5813265245611812,-0.2031240302112651,-1.0205657050683812>, 0.5}
    cylinder { m*<-3.3347679438779165,2.2144617535064204,-1.9199887244990195>, <-1.5813265245611812,-0.2031240302112651,-1.0205657050683812>, 0.5 }
    cylinder {  m*<-2.9808786560202947,-2.7649261992373044,-1.7110151047711164>, <-1.5813265245611812,-0.2031240302112651,-1.0205657050683812>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    