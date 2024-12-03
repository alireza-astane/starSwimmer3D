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
    sphere { m*<-0.31945379583577543,-0.21624822379537578,-0.39749908132899847>, 1 }        
    sphere {  m*<1.0997136983643854,0.7736906900845415,9.451791015706142>, 1 }
    sphere {  m*<8.467500896687174,0.48859843929227886,-5.118886413367781>, 1 }
    sphere {  m*<-6.428462297001811,7.011679812912916,-3.6280795101861747>, 1}
    sphere { m*<-4.39362364475376,-9.088994641234683,-2.2841964123858105>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0997136983643854,0.7736906900845415,9.451791015706142>, <-0.31945379583577543,-0.21624822379537578,-0.39749908132899847>, 0.5 }
    cylinder { m*<8.467500896687174,0.48859843929227886,-5.118886413367781>, <-0.31945379583577543,-0.21624822379537578,-0.39749908132899847>, 0.5}
    cylinder { m*<-6.428462297001811,7.011679812912916,-3.6280795101861747>, <-0.31945379583577543,-0.21624822379537578,-0.39749908132899847>, 0.5 }
    cylinder {  m*<-4.39362364475376,-9.088994641234683,-2.2841964123858105>, <-0.31945379583577543,-0.21624822379537578,-0.39749908132899847>, 0.5}

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
    sphere { m*<-0.31945379583577543,-0.21624822379537578,-0.39749908132899847>, 1 }        
    sphere {  m*<1.0997136983643854,0.7736906900845415,9.451791015706142>, 1 }
    sphere {  m*<8.467500896687174,0.48859843929227886,-5.118886413367781>, 1 }
    sphere {  m*<-6.428462297001811,7.011679812912916,-3.6280795101861747>, 1}
    sphere { m*<-4.39362364475376,-9.088994641234683,-2.2841964123858105>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0997136983643854,0.7736906900845415,9.451791015706142>, <-0.31945379583577543,-0.21624822379537578,-0.39749908132899847>, 0.5 }
    cylinder { m*<8.467500896687174,0.48859843929227886,-5.118886413367781>, <-0.31945379583577543,-0.21624822379537578,-0.39749908132899847>, 0.5}
    cylinder { m*<-6.428462297001811,7.011679812912916,-3.6280795101861747>, <-0.31945379583577543,-0.21624822379537578,-0.39749908132899847>, 0.5 }
    cylinder {  m*<-4.39362364475376,-9.088994641234683,-2.2841964123858105>, <-0.31945379583577543,-0.21624822379537578,-0.39749908132899847>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    