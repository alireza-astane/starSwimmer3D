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
    sphere { m*<-0.4114313882113881,-0.4165574606457124,-0.4400927591792697>, 1 }        
    sphere {  m*<1.0077361059887728,0.5733814532342045,9.409197337855874>, 1 }
    sphere {  m*<8.375523304311573,0.2882892024419428,-5.161480091218053>, 1 }
    sphere {  m*<-6.520439889377424,6.81137057606258,-3.670673188036446>, 1}
    sphere { m*<-3.983704810648626,-8.19627146040972,-2.0943681040590323>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0077361059887728,0.5733814532342045,9.409197337855874>, <-0.4114313882113881,-0.4165574606457124,-0.4400927591792697>, 0.5 }
    cylinder { m*<8.375523304311573,0.2882892024419428,-5.161480091218053>, <-0.4114313882113881,-0.4165574606457124,-0.4400927591792697>, 0.5}
    cylinder { m*<-6.520439889377424,6.81137057606258,-3.670673188036446>, <-0.4114313882113881,-0.4165574606457124,-0.4400927591792697>, 0.5 }
    cylinder {  m*<-3.983704810648626,-8.19627146040972,-2.0943681040590323>, <-0.4114313882113881,-0.4165574606457124,-0.4400927591792697>, 0.5}

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
    sphere { m*<-0.4114313882113881,-0.4165574606457124,-0.4400927591792697>, 1 }        
    sphere {  m*<1.0077361059887728,0.5733814532342045,9.409197337855874>, 1 }
    sphere {  m*<8.375523304311573,0.2882892024419428,-5.161480091218053>, 1 }
    sphere {  m*<-6.520439889377424,6.81137057606258,-3.670673188036446>, 1}
    sphere { m*<-3.983704810648626,-8.19627146040972,-2.0943681040590323>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0077361059887728,0.5733814532342045,9.409197337855874>, <-0.4114313882113881,-0.4165574606457124,-0.4400927591792697>, 0.5 }
    cylinder { m*<8.375523304311573,0.2882892024419428,-5.161480091218053>, <-0.4114313882113881,-0.4165574606457124,-0.4400927591792697>, 0.5}
    cylinder { m*<-6.520439889377424,6.81137057606258,-3.670673188036446>, <-0.4114313882113881,-0.4165574606457124,-0.4400927591792697>, 0.5 }
    cylinder {  m*<-3.983704810648626,-8.19627146040972,-2.0943681040590323>, <-0.4114313882113881,-0.4165574606457124,-0.4400927591792697>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    