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
    sphere { m*<1.2677960750397552,0.02033279750142069,0.6154733783801616>, 1 }        
    sphere {  m*<1.5120432148148195,0.021799401607974063,3.6055136448637333>, 1 }
    sphere {  m*<4.005290403877357,0.02179940160797407,-0.6117685636268828>, 1 }
    sphere {  m*<-3.6557051085311367,8.094713844648826,-2.2956309189743402>, 1}
    sphere { m*<-3.695599819554378,-8.14950970146814,-2.318528264841718>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.5120432148148195,0.021799401607974063,3.6055136448637333>, <1.2677960750397552,0.02033279750142069,0.6154733783801616>, 0.5 }
    cylinder { m*<4.005290403877357,0.02179940160797407,-0.6117685636268828>, <1.2677960750397552,0.02033279750142069,0.6154733783801616>, 0.5}
    cylinder { m*<-3.6557051085311367,8.094713844648826,-2.2956309189743402>, <1.2677960750397552,0.02033279750142069,0.6154733783801616>, 0.5 }
    cylinder {  m*<-3.695599819554378,-8.14950970146814,-2.318528264841718>, <1.2677960750397552,0.02033279750142069,0.6154733783801616>, 0.5}

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
    sphere { m*<1.2677960750397552,0.02033279750142069,0.6154733783801616>, 1 }        
    sphere {  m*<1.5120432148148195,0.021799401607974063,3.6055136448637333>, 1 }
    sphere {  m*<4.005290403877357,0.02179940160797407,-0.6117685636268828>, 1 }
    sphere {  m*<-3.6557051085311367,8.094713844648826,-2.2956309189743402>, 1}
    sphere { m*<-3.695599819554378,-8.14950970146814,-2.318528264841718>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.5120432148148195,0.021799401607974063,3.6055136448637333>, <1.2677960750397552,0.02033279750142069,0.6154733783801616>, 0.5 }
    cylinder { m*<4.005290403877357,0.02179940160797407,-0.6117685636268828>, <1.2677960750397552,0.02033279750142069,0.6154733783801616>, 0.5}
    cylinder { m*<-3.6557051085311367,8.094713844648826,-2.2956309189743402>, <1.2677960750397552,0.02033279750142069,0.6154733783801616>, 0.5 }
    cylinder {  m*<-3.695599819554378,-8.14950970146814,-2.318528264841718>, <1.2677960750397552,0.02033279750142069,0.6154733783801616>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    