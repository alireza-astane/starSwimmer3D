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
    sphere { m*<-0.6718267971367614,-0.9836478170465015,-0.560678630965991>, 1 }        
    sphere {  m*<0.7473406970634012,0.006291096833416443,9.288611466069161>, 1 }
    sphere {  m*<8.115127895386195,-0.27880115395884664,-5.282065963004772>, 1 }
    sphere {  m*<-6.780835298302789,6.2442802196618,-3.791259059823167>, 1}
    sphere { m*<-2.749882858451326,-5.509248161902912,-1.5230005198557897>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7473406970634012,0.006291096833416443,9.288611466069161>, <-0.6718267971367614,-0.9836478170465015,-0.560678630965991>, 0.5 }
    cylinder { m*<8.115127895386195,-0.27880115395884664,-5.282065963004772>, <-0.6718267971367614,-0.9836478170465015,-0.560678630965991>, 0.5}
    cylinder { m*<-6.780835298302789,6.2442802196618,-3.791259059823167>, <-0.6718267971367614,-0.9836478170465015,-0.560678630965991>, 0.5 }
    cylinder {  m*<-2.749882858451326,-5.509248161902912,-1.5230005198557897>, <-0.6718267971367614,-0.9836478170465015,-0.560678630965991>, 0.5}

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
    sphere { m*<-0.6718267971367614,-0.9836478170465015,-0.560678630965991>, 1 }        
    sphere {  m*<0.7473406970634012,0.006291096833416443,9.288611466069161>, 1 }
    sphere {  m*<8.115127895386195,-0.27880115395884664,-5.282065963004772>, 1 }
    sphere {  m*<-6.780835298302789,6.2442802196618,-3.791259059823167>, 1}
    sphere { m*<-2.749882858451326,-5.509248161902912,-1.5230005198557897>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7473406970634012,0.006291096833416443,9.288611466069161>, <-0.6718267971367614,-0.9836478170465015,-0.560678630965991>, 0.5 }
    cylinder { m*<8.115127895386195,-0.27880115395884664,-5.282065963004772>, <-0.6718267971367614,-0.9836478170465015,-0.560678630965991>, 0.5}
    cylinder { m*<-6.780835298302789,6.2442802196618,-3.791259059823167>, <-0.6718267971367614,-0.9836478170465015,-0.560678630965991>, 0.5 }
    cylinder {  m*<-2.749882858451326,-5.509248161902912,-1.5230005198557897>, <-0.6718267971367614,-0.9836478170465015,-0.560678630965991>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    