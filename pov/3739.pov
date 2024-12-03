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
    sphere { m*<-0.006828509862762511,0.19497008362172652,-0.1320068526744324>, 1 }        
    sphere {  m*<0.23390659487892923,0.32368016180205206,2.855547918446119>, 1 }
    sphere {  m*<2.727879884143501,0.29700405900810123,-1.3612163781256195>, 1 }
    sphere {  m*<-1.6284438697556545,2.5234440280403287,-1.105952618090405>, 1}
    sphere { m*<-2.1819790529565455,-3.91684016647464,-1.3922753509482164>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.23390659487892923,0.32368016180205206,2.855547918446119>, <-0.006828509862762511,0.19497008362172652,-0.1320068526744324>, 0.5 }
    cylinder { m*<2.727879884143501,0.29700405900810123,-1.3612163781256195>, <-0.006828509862762511,0.19497008362172652,-0.1320068526744324>, 0.5}
    cylinder { m*<-1.6284438697556545,2.5234440280403287,-1.105952618090405>, <-0.006828509862762511,0.19497008362172652,-0.1320068526744324>, 0.5 }
    cylinder {  m*<-2.1819790529565455,-3.91684016647464,-1.3922753509482164>, <-0.006828509862762511,0.19497008362172652,-0.1320068526744324>, 0.5}

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
    sphere { m*<-0.006828509862762511,0.19497008362172652,-0.1320068526744324>, 1 }        
    sphere {  m*<0.23390659487892923,0.32368016180205206,2.855547918446119>, 1 }
    sphere {  m*<2.727879884143501,0.29700405900810123,-1.3612163781256195>, 1 }
    sphere {  m*<-1.6284438697556545,2.5234440280403287,-1.105952618090405>, 1}
    sphere { m*<-2.1819790529565455,-3.91684016647464,-1.3922753509482164>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.23390659487892923,0.32368016180205206,2.855547918446119>, <-0.006828509862762511,0.19497008362172652,-0.1320068526744324>, 0.5 }
    cylinder { m*<2.727879884143501,0.29700405900810123,-1.3612163781256195>, <-0.006828509862762511,0.19497008362172652,-0.1320068526744324>, 0.5}
    cylinder { m*<-1.6284438697556545,2.5234440280403287,-1.105952618090405>, <-0.006828509862762511,0.19497008362172652,-0.1320068526744324>, 0.5 }
    cylinder {  m*<-2.1819790529565455,-3.91684016647464,-1.3922753509482164>, <-0.006828509862762511,0.19497008362172652,-0.1320068526744324>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    