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
    sphere { m*<-0.5772215856760349,-0.77761613889195,-0.5168681354329142>, 1 }        
    sphere {  m*<0.8419459085241269,0.21232277498796726,9.332421961602234>, 1 }
    sphere {  m*<8.20973310684693,-0.07276947580429405,-5.238255467471696>, 1 }
    sphere {  m*<-6.686230086842061,6.450311897816339,-3.7474485642900888>, 1}
    sphere { m*<-3.2145336650691463,-6.5211668920151205,-1.7381745232057122>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8419459085241269,0.21232277498796726,9.332421961602234>, <-0.5772215856760349,-0.77761613889195,-0.5168681354329142>, 0.5 }
    cylinder { m*<8.20973310684693,-0.07276947580429405,-5.238255467471696>, <-0.5772215856760349,-0.77761613889195,-0.5168681354329142>, 0.5}
    cylinder { m*<-6.686230086842061,6.450311897816339,-3.7474485642900888>, <-0.5772215856760349,-0.77761613889195,-0.5168681354329142>, 0.5 }
    cylinder {  m*<-3.2145336650691463,-6.5211668920151205,-1.7381745232057122>, <-0.5772215856760349,-0.77761613889195,-0.5168681354329142>, 0.5}

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
    sphere { m*<-0.5772215856760349,-0.77761613889195,-0.5168681354329142>, 1 }        
    sphere {  m*<0.8419459085241269,0.21232277498796726,9.332421961602234>, 1 }
    sphere {  m*<8.20973310684693,-0.07276947580429405,-5.238255467471696>, 1 }
    sphere {  m*<-6.686230086842061,6.450311897816339,-3.7474485642900888>, 1}
    sphere { m*<-3.2145336650691463,-6.5211668920151205,-1.7381745232057122>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8419459085241269,0.21232277498796726,9.332421961602234>, <-0.5772215856760349,-0.77761613889195,-0.5168681354329142>, 0.5 }
    cylinder { m*<8.20973310684693,-0.07276947580429405,-5.238255467471696>, <-0.5772215856760349,-0.77761613889195,-0.5168681354329142>, 0.5}
    cylinder { m*<-6.686230086842061,6.450311897816339,-3.7474485642900888>, <-0.5772215856760349,-0.77761613889195,-0.5168681354329142>, 0.5 }
    cylinder {  m*<-3.2145336650691463,-6.5211668920151205,-1.7381745232057122>, <-0.5772215856760349,-0.77761613889195,-0.5168681354329142>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    