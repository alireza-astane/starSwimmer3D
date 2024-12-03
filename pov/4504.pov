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
    sphere { m*<-0.20347791717562547,-0.10400815877075942,-0.8808889862742391>, 1 }        
    sphere {  m*<0.32092338598669246,0.17636529486538227,5.627001156816692>, 1 }
    sphere {  m*<2.5312304768306317,-0.0019741833843852086,-2.11009851172542>, 1 }
    sphere {  m*<-1.8250932770685153,2.2244657856478396,-1.8548347516902068>, 1}
    sphere { m*<-1.5573060560306835,-2.663226156756058,-1.665288466527634>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.32092338598669246,0.17636529486538227,5.627001156816692>, <-0.20347791717562547,-0.10400815877075942,-0.8808889862742391>, 0.5 }
    cylinder { m*<2.5312304768306317,-0.0019741833843852086,-2.11009851172542>, <-0.20347791717562547,-0.10400815877075942,-0.8808889862742391>, 0.5}
    cylinder { m*<-1.8250932770685153,2.2244657856478396,-1.8548347516902068>, <-0.20347791717562547,-0.10400815877075942,-0.8808889862742391>, 0.5 }
    cylinder {  m*<-1.5573060560306835,-2.663226156756058,-1.665288466527634>, <-0.20347791717562547,-0.10400815877075942,-0.8808889862742391>, 0.5}

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
    sphere { m*<-0.20347791717562547,-0.10400815877075942,-0.8808889862742391>, 1 }        
    sphere {  m*<0.32092338598669246,0.17636529486538227,5.627001156816692>, 1 }
    sphere {  m*<2.5312304768306317,-0.0019741833843852086,-2.11009851172542>, 1 }
    sphere {  m*<-1.8250932770685153,2.2244657856478396,-1.8548347516902068>, 1}
    sphere { m*<-1.5573060560306835,-2.663226156756058,-1.665288466527634>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.32092338598669246,0.17636529486538227,5.627001156816692>, <-0.20347791717562547,-0.10400815877075942,-0.8808889862742391>, 0.5 }
    cylinder { m*<2.5312304768306317,-0.0019741833843852086,-2.11009851172542>, <-0.20347791717562547,-0.10400815877075942,-0.8808889862742391>, 0.5}
    cylinder { m*<-1.8250932770685153,2.2244657856478396,-1.8548347516902068>, <-0.20347791717562547,-0.10400815877075942,-0.8808889862742391>, 0.5 }
    cylinder {  m*<-1.5573060560306835,-2.663226156756058,-1.665288466527634>, <-0.20347791717562547,-0.10400815877075942,-0.8808889862742391>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    