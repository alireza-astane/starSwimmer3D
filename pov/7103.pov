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
    sphere { m*<-0.7600882312023673,-1.175863986706596,-0.6015514021714063>, 1 }        
    sphere {  m*<0.6590792629977952,-0.18592507282667836,9.247738694863747>, 1 }
    sphere {  m*<8.026866461320601,-0.4710173236189402,-5.322938734210189>, 1 }
    sphere {  m*<-6.8690967323684,6.052064050001713,-3.8321318310285832>, 1}
    sphere { m*<-2.2874922131780564,-4.502251621884967,-1.3088731690840985>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6590792629977952,-0.18592507282667836,9.247738694863747>, <-0.7600882312023673,-1.175863986706596,-0.6015514021714063>, 0.5 }
    cylinder { m*<8.026866461320601,-0.4710173236189402,-5.322938734210189>, <-0.7600882312023673,-1.175863986706596,-0.6015514021714063>, 0.5}
    cylinder { m*<-6.8690967323684,6.052064050001713,-3.8321318310285832>, <-0.7600882312023673,-1.175863986706596,-0.6015514021714063>, 0.5 }
    cylinder {  m*<-2.2874922131780564,-4.502251621884967,-1.3088731690840985>, <-0.7600882312023673,-1.175863986706596,-0.6015514021714063>, 0.5}

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
    sphere { m*<-0.7600882312023673,-1.175863986706596,-0.6015514021714063>, 1 }        
    sphere {  m*<0.6590792629977952,-0.18592507282667836,9.247738694863747>, 1 }
    sphere {  m*<8.026866461320601,-0.4710173236189402,-5.322938734210189>, 1 }
    sphere {  m*<-6.8690967323684,6.052064050001713,-3.8321318310285832>, 1}
    sphere { m*<-2.2874922131780564,-4.502251621884967,-1.3088731690840985>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6590792629977952,-0.18592507282667836,9.247738694863747>, <-0.7600882312023673,-1.175863986706596,-0.6015514021714063>, 0.5 }
    cylinder { m*<8.026866461320601,-0.4710173236189402,-5.322938734210189>, <-0.7600882312023673,-1.175863986706596,-0.6015514021714063>, 0.5}
    cylinder { m*<-6.8690967323684,6.052064050001713,-3.8321318310285832>, <-0.7600882312023673,-1.175863986706596,-0.6015514021714063>, 0.5 }
    cylinder {  m*<-2.2874922131780564,-4.502251621884967,-1.3088731690840985>, <-0.7600882312023673,-1.175863986706596,-0.6015514021714063>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    