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
    sphere { m*<-0.29725455699570835,-0.13938595779425092,-1.6630185573241447>, 1 }        
    sphere {  m*<0.5227468436807587,0.2905507303554801,8.294027361486553>, 1 }
    sphere {  m*<2.6074499929901918,-0.031246837263669933,-2.974780627067119>, 1 }
    sphere {  m*<-1.9204227714897129,2.189104792919532,-2.6343337009706302>, 1}
    sphere { m*<-1.652635550451881,-2.698587149484365,-2.44478741580806>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5227468436807587,0.2905507303554801,8.294027361486553>, <-0.29725455699570835,-0.13938595779425092,-1.6630185573241447>, 0.5 }
    cylinder { m*<2.6074499929901918,-0.031246837263669933,-2.974780627067119>, <-0.29725455699570835,-0.13938595779425092,-1.6630185573241447>, 0.5}
    cylinder { m*<-1.9204227714897129,2.189104792919532,-2.6343337009706302>, <-0.29725455699570835,-0.13938595779425092,-1.6630185573241447>, 0.5 }
    cylinder {  m*<-1.652635550451881,-2.698587149484365,-2.44478741580806>, <-0.29725455699570835,-0.13938595779425092,-1.6630185573241447>, 0.5}

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
    sphere { m*<-0.29725455699570835,-0.13938595779425092,-1.6630185573241447>, 1 }        
    sphere {  m*<0.5227468436807587,0.2905507303554801,8.294027361486553>, 1 }
    sphere {  m*<2.6074499929901918,-0.031246837263669933,-2.974780627067119>, 1 }
    sphere {  m*<-1.9204227714897129,2.189104792919532,-2.6343337009706302>, 1}
    sphere { m*<-1.652635550451881,-2.698587149484365,-2.44478741580806>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5227468436807587,0.2905507303554801,8.294027361486553>, <-0.29725455699570835,-0.13938595779425092,-1.6630185573241447>, 0.5 }
    cylinder { m*<2.6074499929901918,-0.031246837263669933,-2.974780627067119>, <-0.29725455699570835,-0.13938595779425092,-1.6630185573241447>, 0.5}
    cylinder { m*<-1.9204227714897129,2.189104792919532,-2.6343337009706302>, <-0.29725455699570835,-0.13938595779425092,-1.6630185573241447>, 0.5 }
    cylinder {  m*<-1.652635550451881,-2.698587149484365,-2.44478741580806>, <-0.29725455699570835,-0.13938595779425092,-1.6630185573241447>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    