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
    sphere { m*<-1.5167211180937603,-0.18308389479221154,-1.0683126938983287>, 1 }        
    sphere {  m*<-0.08064638791543552,0.2777841829410627,8.817254682305656>, 1 }
    sphere {  m*<6.992850218587627,0.10842984982048748,-5.565108209282249>, 1 }
    sphere {  m*<-3.191588167981351,2.1462270752187598,-1.945306557181714>, 1}
    sphere { m*<-2.9238009469435196,-2.7414648671851376,-1.7557602720191436>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.08064638791543552,0.2777841829410627,8.817254682305656>, <-1.5167211180937603,-0.18308389479221154,-1.0683126938983287>, 0.5 }
    cylinder { m*<6.992850218587627,0.10842984982048748,-5.565108209282249>, <-1.5167211180937603,-0.18308389479221154,-1.0683126938983287>, 0.5}
    cylinder { m*<-3.191588167981351,2.1462270752187598,-1.945306557181714>, <-1.5167211180937603,-0.18308389479221154,-1.0683126938983287>, 0.5 }
    cylinder {  m*<-2.9238009469435196,-2.7414648671851376,-1.7557602720191436>, <-1.5167211180937603,-0.18308389479221154,-1.0683126938983287>, 0.5}

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
    sphere { m*<-1.5167211180937603,-0.18308389479221154,-1.0683126938983287>, 1 }        
    sphere {  m*<-0.08064638791543552,0.2777841829410627,8.817254682305656>, 1 }
    sphere {  m*<6.992850218587627,0.10842984982048748,-5.565108209282249>, 1 }
    sphere {  m*<-3.191588167981351,2.1462270752187598,-1.945306557181714>, 1}
    sphere { m*<-2.9238009469435196,-2.7414648671851376,-1.7557602720191436>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.08064638791543552,0.2777841829410627,8.817254682305656>, <-1.5167211180937603,-0.18308389479221154,-1.0683126938983287>, 0.5 }
    cylinder { m*<6.992850218587627,0.10842984982048748,-5.565108209282249>, <-1.5167211180937603,-0.18308389479221154,-1.0683126938983287>, 0.5}
    cylinder { m*<-3.191588167981351,2.1462270752187598,-1.945306557181714>, <-1.5167211180937603,-0.18308389479221154,-1.0683126938983287>, 0.5 }
    cylinder {  m*<-2.9238009469435196,-2.7414648671851376,-1.7557602720191436>, <-1.5167211180937603,-0.18308389479221154,-1.0683126938983287>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    