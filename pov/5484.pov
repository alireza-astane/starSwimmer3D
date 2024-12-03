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
    sphere { m*<-0.8772039530813502,-0.1612497993968759,-1.4081709669221232>, 1 }        
    sphere {  m*<0.26492810918897497,0.28504740051422184,8.516340687824552>, 1 }
    sphere {  m*<4.8599853733349905,0.04357130648300453,-4.225803539332019>, 1 }
    sphere {  m*<-2.5280882745325193,2.167613754819037,-2.33065018340287>, 1}
    sphere { m*<-2.260301053494688,-2.7200781875848605,-2.1411038982402997>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.26492810918897497,0.28504740051422184,8.516340687824552>, <-0.8772039530813502,-0.1612497993968759,-1.4081709669221232>, 0.5 }
    cylinder { m*<4.8599853733349905,0.04357130648300453,-4.225803539332019>, <-0.8772039530813502,-0.1612497993968759,-1.4081709669221232>, 0.5}
    cylinder { m*<-2.5280882745325193,2.167613754819037,-2.33065018340287>, <-0.8772039530813502,-0.1612497993968759,-1.4081709669221232>, 0.5 }
    cylinder {  m*<-2.260301053494688,-2.7200781875848605,-2.1411038982402997>, <-0.8772039530813502,-0.1612497993968759,-1.4081709669221232>, 0.5}

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
    sphere { m*<-0.8772039530813502,-0.1612497993968759,-1.4081709669221232>, 1 }        
    sphere {  m*<0.26492810918897497,0.28504740051422184,8.516340687824552>, 1 }
    sphere {  m*<4.8599853733349905,0.04357130648300453,-4.225803539332019>, 1 }
    sphere {  m*<-2.5280882745325193,2.167613754819037,-2.33065018340287>, 1}
    sphere { m*<-2.260301053494688,-2.7200781875848605,-2.1411038982402997>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.26492810918897497,0.28504740051422184,8.516340687824552>, <-0.8772039530813502,-0.1612497993968759,-1.4081709669221232>, 0.5 }
    cylinder { m*<4.8599853733349905,0.04357130648300453,-4.225803539332019>, <-0.8772039530813502,-0.1612497993968759,-1.4081709669221232>, 0.5}
    cylinder { m*<-2.5280882745325193,2.167613754819037,-2.33065018340287>, <-0.8772039530813502,-0.1612497993968759,-1.4081709669221232>, 0.5 }
    cylinder {  m*<-2.260301053494688,-2.7200781875848605,-2.1411038982402997>, <-0.8772039530813502,-0.1612497993968759,-1.4081709669221232>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    