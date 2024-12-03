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
    sphere { m*<-0.7345154296095368,-1.1201714181267823,-0.5897089561512305>, 1 }        
    sphere {  m*<0.6846520645906254,-0.13023250424686483,9.25958114088392>, 1 }
    sphere {  m*<8.052439262913426,-0.41532475503912747,-5.311096288190012>, 1 }
    sphere {  m*<-6.843523930775564,6.107756618581526,-3.820289385008408>, 1}
    sphere { m*<-2.425394247638954,-4.802575326490822,-1.3727338842758767>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6846520645906254,-0.13023250424686483,9.25958114088392>, <-0.7345154296095368,-1.1201714181267823,-0.5897089561512305>, 0.5 }
    cylinder { m*<8.052439262913426,-0.41532475503912747,-5.311096288190012>, <-0.7345154296095368,-1.1201714181267823,-0.5897089561512305>, 0.5}
    cylinder { m*<-6.843523930775564,6.107756618581526,-3.820289385008408>, <-0.7345154296095368,-1.1201714181267823,-0.5897089561512305>, 0.5 }
    cylinder {  m*<-2.425394247638954,-4.802575326490822,-1.3727338842758767>, <-0.7345154296095368,-1.1201714181267823,-0.5897089561512305>, 0.5}

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
    sphere { m*<-0.7345154296095368,-1.1201714181267823,-0.5897089561512305>, 1 }        
    sphere {  m*<0.6846520645906254,-0.13023250424686483,9.25958114088392>, 1 }
    sphere {  m*<8.052439262913426,-0.41532475503912747,-5.311096288190012>, 1 }
    sphere {  m*<-6.843523930775564,6.107756618581526,-3.820289385008408>, 1}
    sphere { m*<-2.425394247638954,-4.802575326490822,-1.3727338842758767>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6846520645906254,-0.13023250424686483,9.25958114088392>, <-0.7345154296095368,-1.1201714181267823,-0.5897089561512305>, 0.5 }
    cylinder { m*<8.052439262913426,-0.41532475503912747,-5.311096288190012>, <-0.7345154296095368,-1.1201714181267823,-0.5897089561512305>, 0.5}
    cylinder { m*<-6.843523930775564,6.107756618581526,-3.820289385008408>, <-0.7345154296095368,-1.1201714181267823,-0.5897089561512305>, 0.5 }
    cylinder {  m*<-2.425394247638954,-4.802575326490822,-1.3727338842758767>, <-0.7345154296095368,-1.1201714181267823,-0.5897089561512305>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    