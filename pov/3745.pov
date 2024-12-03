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
    sphere { m*<-0.010481147556592463,0.18806529574069253,-0.13412316790408724>, 1 }        
    sphere {  m*<0.23025395718509928,0.31677537392101796,2.8534316032164644>, 1 }
    sphere {  m*<2.7242272464496713,0.29009927112706724,-1.3633326933552743>, 1 }
    sphere {  m*<-1.6320965074494844,2.516539240159295,-1.1080689333200593>, 1}
    sphere { m*<-2.1666780967064043,-3.887915902383879,-1.3834100734543235>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.23025395718509928,0.31677537392101796,2.8534316032164644>, <-0.010481147556592463,0.18806529574069253,-0.13412316790408724>, 0.5 }
    cylinder { m*<2.7242272464496713,0.29009927112706724,-1.3633326933552743>, <-0.010481147556592463,0.18806529574069253,-0.13412316790408724>, 0.5}
    cylinder { m*<-1.6320965074494844,2.516539240159295,-1.1080689333200593>, <-0.010481147556592463,0.18806529574069253,-0.13412316790408724>, 0.5 }
    cylinder {  m*<-2.1666780967064043,-3.887915902383879,-1.3834100734543235>, <-0.010481147556592463,0.18806529574069253,-0.13412316790408724>, 0.5}

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
    sphere { m*<-0.010481147556592463,0.18806529574069253,-0.13412316790408724>, 1 }        
    sphere {  m*<0.23025395718509928,0.31677537392101796,2.8534316032164644>, 1 }
    sphere {  m*<2.7242272464496713,0.29009927112706724,-1.3633326933552743>, 1 }
    sphere {  m*<-1.6320965074494844,2.516539240159295,-1.1080689333200593>, 1}
    sphere { m*<-2.1666780967064043,-3.887915902383879,-1.3834100734543235>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.23025395718509928,0.31677537392101796,2.8534316032164644>, <-0.010481147556592463,0.18806529574069253,-0.13412316790408724>, 0.5 }
    cylinder { m*<2.7242272464496713,0.29009927112706724,-1.3633326933552743>, <-0.010481147556592463,0.18806529574069253,-0.13412316790408724>, 0.5}
    cylinder { m*<-1.6320965074494844,2.516539240159295,-1.1080689333200593>, <-0.010481147556592463,0.18806529574069253,-0.13412316790408724>, 0.5 }
    cylinder {  m*<-2.1666780967064043,-3.887915902383879,-1.3834100734543235>, <-0.010481147556592463,0.18806529574069253,-0.13412316790408724>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    